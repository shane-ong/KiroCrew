// The quit/update stop path when the gateway child is a launcher shim that
// FORKED the real gateway instead of exec'ing it. SIGTERM reaches only the
// shim; without the tree listing the gateway lives on re-parented to init,
// holding the port and the lock.
const { test } = require("node:test");
const assert = require("node:assert");
const { stopGatewayGracefully } = require("../gateway-stop");

const GATEWAY_CMD = "/opt/py/bin/python3 -m kirocrew_edition gateway --port 5476";

// A fake process table. `procs[pid] = { cmd, onSignal(signal) -> "die" | "live" }`.
function world(procs) {
  const sent = [];
  const alive = new Set(Object.keys(procs).map(Number));
  const signalPidFn = (pid, signal) => {
    if (!alive.has(pid)) { const e = new Error("ESRCH"); e.code = "ESRCH"; throw e; }
    if (signal === 0) return;
    sent.push([pid, signal]);
    if (signal === "SIGKILL" || procs[pid].onSignal(signal) === "die") alive.delete(pid);
  };
  return {
    sent,
    alive,
    signalPidFn,
    getCommandFn: async (pid) => (alive.has(pid) ? procs[pid].cmd : ""),
  };
}

// The shim: the direct child. It dies on any signal.
function shimProc(pid = 100) {
  return {
    pid,
    exitCode: null,
    _onExit: [],
    signals: [],
    kill(sig) {
      this.signals.push(sig);
      this.exitCode = 0;
      for (const fn of this._onExit) fn();
    },
    once(event, fn) { if (event === "exit") this._onExit.push(fn); },
  };
}

const base = {
  backendUrl: "http://127.0.0.1:1",
  kirocrewHome: "/nope",
  postShutdownFn: async () => false, // the endpoint failed: the signal path
  platform: "darwin",
  survivorPollMs: 5,
};

test("shim: the gateway it forked gets SIGTERM once the shim is gone", async () => {
  const w = world({ 101: { cmd: GATEWAY_CMD, onSignal: () => "die" } });
  const proc = shimProc();
  await stopGatewayGracefully(proc, {
    ...base,
    timeoutMs: 2000,
    listDescendantsFn: async () => [101],
    getCommandFn: w.getCommandFn,
    signalPidFn: w.signalPidFn,
  });
  assert.deepStrictEqual(proc.signals, ["SIGTERM"], "the shim itself still gets one SIGTERM");
  assert.deepStrictEqual(w.sent, [[101, "SIGTERM"]], "graceful first, and no SIGKILL once it exits");
  assert.ok(!w.alive.has(101));
});

test("shim: a forked gateway that ignores SIGTERM is SIGKILLed at the deadline", async () => {
  const w = world({ 101: { cmd: GATEWAY_CMD, onSignal: () => "live" } });
  const started = Date.now();
  await stopGatewayGracefully(shimProc(), {
    ...base,
    timeoutMs: 150,
    listDescendantsFn: async () => [101],
    getCommandFn: w.getCommandFn,
    signalPidFn: w.signalPidFn,
  });
  assert.deepStrictEqual(w.sent, [[101, "SIGTERM"], [101, "SIGKILL"]]);
  assert.ok(Date.now() - started >= 140, "SIGKILL waits for the deadline, not sooner");
});

test("shim: a leftover that is not a gateway is left for the backend's orphan sweep", async () => {
  const w = world({ 102: { cmd: "/usr/local/bin/kiro-cli acp", onSignal: () => "die" } });
  await stopGatewayGracefully(shimProc(), {
    ...base,
    timeoutMs: 100,
    listDescendantsFn: async () => [102],
    getCommandFn: w.getCommandFn,
    signalPidFn: w.signalPidFn,
  });
  assert.deepStrictEqual(w.sent, []);
});

test("shim: a survivor whose pid stops reading as a gateway is never SIGKILLed", async () => {
  // The gateway ignores SIGTERM, then (the pid reused) reads as another program.
  let cmd = GATEWAY_CMD;
  const w = world({ 101: { get cmd() { return cmd; }, onSignal: () => { cmd = "/bin/bash"; return "live"; } } });
  await stopGatewayGracefully(shimProc(), {
    ...base,
    timeoutMs: 100,
    listDescendantsFn: async () => [101],
    getCommandFn: w.getCommandFn,
    signalPidFn: w.signalPidFn,
  });
  assert.deepStrictEqual(w.sent, [[101, "SIGTERM"]], "identity is re-read before SIGKILL");
});

test("a gateway child that outlives the deadline is SIGKILLed with its whole tree", async () => {
  const w = world({ 201: { cmd: "kiro-cli acp", onSignal: () => "live" } });
  const listed = [];
  const proc = {
    pid: 200,
    exitCode: null,
    _onExit: [],
    signals: [],
    kill(sig) {
      this.signals.push(sig);
      if (sig !== "SIGKILL") return; // ignores SIGTERM: wedged
      this.exitCode = 1;
      for (const fn of this._onExit) fn();
    },
    once(event, fn) { if (event === "exit") this._onExit.push(fn); },
  };
  await stopGatewayGracefully(proc, {
    ...base,
    timeoutMs: 60,
    listDescendantsFn: async (pid) => { listed.push(pid); return [201]; },
    getCommandFn: w.getCommandFn,
    signalPidFn: w.signalPidFn,
  });
  assert.deepStrictEqual(proc.signals, ["SIGTERM", "SIGKILL"]);
  assert.deepStrictEqual(listed, [200, 200], "listed before each signal");
  assert.deepStrictEqual(w.sent, [[201, "SIGKILL"]], "SIGTERM stays on the child; SIGKILL takes the tree");
});

test("a listing that fails still signals the child", async () => {
  const proc = shimProc();
  await stopGatewayGracefully(proc, {
    ...base,
    timeoutMs: 100,
    listDescendantsFn: async () => { throw new Error("ps missing"); },
    getCommandFn: async () => "",
    signalPidFn: () => {},
  });
  assert.deepStrictEqual(proc.signals, ["SIGTERM"]);
});

test("shim: each pid's SIGTERM follows its own identity check, with no other check between", async () => {
  // Two listed gateways. The trace must read check(101), TERM(101), check(102), TERM(102):
  // a pid checked first and signalled after other ps reads could be reused in that gap.
  const trace = [];
  const w = world({
    101: { cmd: GATEWAY_CMD, onSignal: () => "die" },
    102: { cmd: GATEWAY_CMD, onSignal: () => "die" },
  });
  await stopGatewayGracefully(shimProc(), {
    ...base,
    timeoutMs: 500,
    listDescendantsFn: async () => [101, 102],
    getCommandFn: async (pid) => { trace.push(`check:${pid}`); return w.getCommandFn(pid); },
    signalPidFn: (pid, sig) => { if (sig !== 0) trace.push(`${sig}:${pid}`); return w.signalPidFn(pid, sig); },
  });
  assert.deepStrictEqual(trace, ["check:101", "SIGTERM:101", "check:102", "SIGTERM:102"]);
});
