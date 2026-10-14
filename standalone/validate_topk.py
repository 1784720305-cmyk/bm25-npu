"""用现有NPU调用器补验K=1、K=文章数和重复运行，失败即停止并保存报告。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.run_topk import prepare_command
from standalone.topk_fixture import check, write_input, MAX_K
from check_result import read_floats


def save_report(path, report):
    path.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def execute(command, log_path):
    """Stream the actual caller's stages to console and a dedicated log, without shell expansion."""
    print('实际NPU命令：' + json.dumps(command, ensure_ascii=False), flush=True)
    with log_path.open('w', encoding='utf-8', newline='\n') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            # Terminate only this script's own child if user interruption stops log collection.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def compare_repeat(previous, current):
    hashes = {}
    for name in ('top_scores.bin', 'top_ids.bin'):
        old, new = (previous / name).read_bytes(), (current / name).read_bytes()
        if old != new:
            raise ValueError(f'重复运行文件不一致：{name}')
        hashes[name] = hashlib.sha256(new).hexdigest()
    return hashes


def run_suite(directory, previous, output, runner, device):
    directory, previous, output, runner = (p.resolve() for p in (directory, previous, output, runner))
    # Check source metadata and runner before creating any new report/input directories.
    _, info = prepare_command(directory, output, runner, device)
    if info['num_docs'] > MAX_K:
        raise ValueError('此补验脚本要求文章数<=32，以便验证K=文章数；更大输入使用单独边界用例')
    previous_info = json.loads((previous / 'run_metadata.json').read_text(encoding='utf-8'))
    if any(previous_info.get(key) != info[key] for key in ('num_queries', 'num_docs', 'k')):
        raise ValueError('此前输出的尺寸/K与当前输入不同，不能作重复运行对比')
    if previous_info.get('input_sha256') != info['scores_sha256']:
        raise ValueError('此前输出使用的输入指纹不同')
    check(directory, previous / 'top_scores.bin', previous / 'top_ids.bin')
    scores = read_floats(directory / 'scores.bin')
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'report.json'
    report = {
        'status': 'running', 'device': device, 'source_input': str(directory),
        'source_sha256': info['scores_sha256'], 'previous_output': str(previous),
        'scope': 'Top-K boundary correctness and independent-process repeat only; not a performance benchmark',
        'cases': [],
    }
    save_report(report_path, report)
    cases = [('k1', 1), ('k_all_docs', info['num_docs']), ('repeat_original_k', info['k'])]
    try:
        for name, k in cases:
            print(f'开始{name}：查询数={info["num_queries"]}，文章数={info["num_docs"]}，K={k}', flush=True)
            case = {'name': name, 'k': k, 'status': 'running'}
            report['cases'].append(case)
            save_report(report_path, report)
            case_dir = output / name
            case_dir.mkdir()
            input_dir, actual_dir = case_dir / 'input', case_dir / 'actual'
            write_input(scores, info['num_queries'], info['num_docs'], k, input_dir,
                        info['input_origin'], str(directory / 'scores.bin'))
            command, _ = prepare_command(input_dir, actual_dir, runner, device)
            case['command'] = command
            case['run_log'] = str(case_dir / 'run.log')
            save_report(report_path, report)
            execute(command, case_dir / 'run.log')
            with (case_dir / 'check.log').open('w', encoding='utf-8', newline='\n') as check_log:
                with redirect_stdout(check_log):
                    check(input_dir, actual_dir / 'top_scores.bin', actual_dir / 'top_ids.bin')
            if name == 'repeat_original_k':
                case['repeat_sha256'] = compare_repeat(previous, actual_dir)
            case['status'] = 'passed'
            save_report(report_path, report)
            print(f'{name}通过：实际设备运行、编号和分数字节核对通过。', flush=True)
        report['status'] = 'passed'
        save_report(report_path, report)
        print(f'三项补验全部通过：K=1、K=全部文章、重复运行字节一致。报告：{report_path}', flush=True)
        print('这是Top-K正确性补验，不是同进程复用验证，也不证明BM25性能达标。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or type(error).__name__
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            report['cases'][-1]['status'] = report['status']
            report['cases'][-1]['error'] = report['error']
        save_report(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(ROOT / 'results/topk_input'))
    parser.add_argument('--previous', default=str(ROOT / 'results/topk_run1'))
    parser.add_argument('--output', required=True, help='新目录，保存各用例输入/设备输出/日志和report.json')
    parser.add_argument('--runner', default=str(ROOT / 'results/aclnn_topk_host_build/bm25_topk_runner'))
    parser.add_argument('--device', type=int, default=2)
    args = parser.parse_args()
    try:
        run_suite(Path(args.input), Path(args.previous), Path(args.output), Path(args.runner), args.device)
    except KeyboardInterrupt:
        print('补验已中断；保留已完成报告，不将未完成用例记为通过。', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'FAILED：{error}；停止，不使用CPU替代输出。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
