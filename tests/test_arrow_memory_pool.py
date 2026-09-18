"""Exercise Arrow thread lifetimes in a fresh process: SIGSEGV is not catchable."""

import os
import subprocess
import sys
import textwrap


def test_parquet_survives_first_import_thread_exiting(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("ARROW_DEFAULT_MEMORY_POOL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            textwrap.dedent("""\
                import sys
                import threading
                from concurrent.futures import Future
                from pathlib import Path

                import flexivtrainer

                def parquet_roundtrip(index):
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    assert pa.default_memory_pool().backend_name == "system"
                    table = pa.table({"action": [[1.0, 2.0]] * 64})
                    path = Path(sys.argv[1]) / f"{index}.parquet"
                    pq.write_table(table, path)
                    assert pq.read_table(path).equals(table)

                def run(result, index):
                    try:
                        parquet_roundtrip(index)
                    except BaseException as exc:
                        result.set_exception(exc)
                    else:
                        result.set_result(None)

                # First import/allocation happens on a worker that exits;
                # subsequent fresh workers must still be able to allocate.
                for index in range(20):
                    result = Future()
                    thread = threading.Thread(target=run, args=(result, index))
                    thread.start()
                    thread.join()
                    result.result()
                """),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_arrow_allocator_is_preserved() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import flexivtrainer; import os; "
            "assert os.environ['ARROW_DEFAULT_MEMORY_POOL'] == 'jemalloc'",
        ],
        env={**os.environ, "ARROW_DEFAULT_MEMORY_POOL": "jemalloc"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
