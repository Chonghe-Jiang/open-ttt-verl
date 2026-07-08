import json
import subprocess
from pathlib import Path


def test_local_gojudge_handles_child_stdin_stream_errors():
    source = Path("scripts/modal_local_gojudge.js").read_text()

    assert "child.stdin.on('error'" in source


def test_local_gojudge_does_not_crash_when_child_closes_stdin_early(tmp_path):
    module_path = (Path.cwd() / "scripts" / "modal_local_gojudge.js").as_posix()
    script_path = tmp_path / "repro_epipe.mjs"
    script_path.write_text(
        f"""
import {{ GoJudgeClient }} from {json.dumps(module_path)};

const client = new GoJudgeClient();
const result = await client.runOne({{
  args: ["/usr/bin/python3", "-c", "import os, time; os.close(0); time.sleep(0.2)"],
  env: ["PATH=/usr/bin:/bin"],
  files: [
    {{ content: "x".repeat(64 * 1024 * 1024) }},
    {{ name: "stdout", max: 1024 }},
    {{ name: "stderr", max: 1024 }}
  ],
  cpuLimit: 10e9,
  clockLimit: 20e9,
  memoryLimit: 256 << 20,
  procLimit: 10
}});
console.log(JSON.stringify(result));
""".strip()
    )

    completed = subprocess.run(
        ["node", str(script_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "Accepted"
