"""Resource-only build of a verified copy. Never install/run its output or modify the accepted project."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import run_fused_chain as fused, resident_index as r

EXCLUDED={'build_out','build','build_host','build_device','autogen','.git','bm25_template_backup','__pycache__'}
LOGGING_REFERENCE='https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/asc-devkit/docs/zh/guide/programming_guide/advanced_programming/aclnn_operator_development/compilation_and_deployment/basic_process.md'


def cache(path):
    values={}
    if path.is_file():
        for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
            found=re.match(r'([^:#]+):[^=]+=(.*)',line)
            if found:values[found[1]]=found[2]
    return {key:value for key,value in values.items() if key in ('CMAKE_BUILD_TYPE','ASCEND_COMPUTE_UNIT','vendor_name')
        or key.startswith(('CMAKE_CXX_FLAGS','CMAKE_C_FLAGS'))}


def resource_lines(text):
    lines=text.splitlines();selected=set()
    pattern=re.compile(r'FusedScoreTopKSimt|Function properties for|Stack[_\s]*size|register(?:s|\s+count|\s+number|\s+size|\s+usage)|\b(?:GPR|VGPR|SGPR|RegNum|UsedRegs)\b|spill',re.I)
    for i,line in enumerate(lines):
        if pattern.search(line):selected.update(range(max(0,i-2),min(len(lines),i+4)))
    return [lines[i][:1000] for i in sorted(selected)][:120]


def compiler_reports(output):
    """Read saved stage logs, including *_success files without a .log suffix; missing fields stay UNKNOWN."""
    stage=output/'compiler_logs'
    paths=[output/'resource_build.log']+sorted(p for p in stage.rglob('*') if p.is_file())
    records=[];vf=[]
    pattern=re.compile(r'Function properties for\s+([^:\n]+):\s*Stack\s+size:\s*(\d+)\s+bytes'
        r'(?:,[ \t]*Used register number:[ \t]*(\d+))?',re.I)
    for path in paths:
        r.require(not path.is_symlink() and path.resolve().is_relative_to(output.resolve()),'资源日志路径越界或包含符号链接')
        size=path.stat().st_size
        item={'path':path.relative_to(output).as_posix(),'bytes':size,'sha256':fused.digest(path)}
        if size>16*1024*1024:
            records.append(dict(item,unread_reason='Log exceeds 16MiB; fields UNKNOWN, inspect this saved file separately.'));continue
        data=path.read_bytes()
        try:text=data.decode('utf-8')
        except UnicodeError:
            records.append(dict(item,unread_reason='Not UTF-8 text; fields UNKNOWN.'));continue
        if '\0' in text:
            records.append(dict(item,unread_reason='Binary data; not interpreted as compiler report.'));continue
        records.append(dict(item,resource_context_lines=resource_lines(text)))
        for match in pattern.finditer(text):
            if 'FusedScoreTopKSimt' not in match[1]:continue
            vf.append({'path':item['path'],'function':match[1],'stack_size_bytes':int(match[2]),
                'used_register_number':int(match[3]) if match[3] is not None else None})
    return records,vf


def prepare(project,target):
    """Require the actual generated Debug hook signature before adding its known OPTIONS argument."""
    project=project.resolve(strict=True);target=target.resolve()
    r.require(not target.exists() and target.is_relative_to((ROOT/'results').resolve()) and
        target!=project and not target.is_relative_to(project) and not project.is_relative_to(target),
        '资源副本必须是results下新独立目录，不能覆盖/嵌套原工程')
    cmake=project/'op_kernel/CMakeLists.txt';text=cmake.read_text(encoding='utf-8')
    # CANN 9.2 target template supplied from the actual A5 project. Keep the older known hook too.
    prefix=r'(?im)^[ \t]*if\s*\([^\n]*CMAKE_BUILD_TYPE[^\n]*Debug[^\n]*\)\s*\n[ \t]*'
    hooks=[]
    for api,args in (('npu_op_kernel_options',r'ascendc_kernels\s+ALL\s+OPTIONS'),
                     ('add_ops_compile_options',r'ALL\s+OPTIONS')):
        for match in re.finditer(prefix+api+r'\s*\(\s*'+args+r'\b[^)]*\)\s*endif\s*\(\s*\)',text):
            hooks.append((match,api))
    if len(hooks)!=1:return {'status':'unsupported_actual_template','actual_cmake_sha256':fused.digest(cmake),
        'reason':'实际模板没有唯一已核对的Debug选项钩子；仅支持npu_op_kernel_options(ascendc_kernels ALL OPTIONS ...)或旧版add_ops_compile_options；不猜API/目标'}
    hook,api=hooks[0]
    r.require('--cce-res-usage' not in text,'实际模板已含资源选项，不重复插入')
    arguments='ascendc_kernels ALL OPTIONS' if api=='npu_op_kernel_options' else 'ALL OPTIONS'
    for directory,dirs,files in os.walk(project,followlinks=False):
        dirs[:]=[name for name in dirs if name not in EXCLUDED]
        for name in dirs+files:r.require(not (Path(directory)/name).is_symlink(),'资源副本拒绝符号链接：'+name)
    def ignored(unused,names):return [name for name in names if name in EXCLUDED]
    shutil.copytree(project,target,ignore=ignored)
    for name in fused.generator.SOURCES+('CMakePresets.json','build.sh'):
        r.require(fused.digest(project/name)==fused.digest(target/name),'资源副本源码/配置不同：'+name)
    # Stop if generated build scripts reference the accepted directory by absolute path.
    for source in target.rglob('*'):
        if source.is_file() and (source.suffix in ('.sh','.cmake') or source.name=='CMakeLists.txt'):
            r.require(str(project) not in source.read_text(encoding='utf-8',errors='replace'),
                '副本构建脚本硬编码原工程路径；停止，不能在原目录构建')
    # Outside the Debug condition: preserve Release optimization and the existing Debug block byte-for-byte.
    added='\n# Resource diagnostic COPY ONLY; no precision/optimization/thread changes.\n'+api+'('+arguments+' --cce-res-usage)\n'
    (target/'op_kernel/CMakeLists.txt').write_bytes((text[:hook.start()]+added+text[hook.start():]).encode('utf-8'))
    return {'status':'prepared_resource_only_copy','actual_cmake_sha256':fused.digest(cmake),
        'copy_cmake_sha256':fused.digest(target/'op_kernel/CMakeLists.txt'),
        'source_sha256':{name:fused.digest(target/name) for name in fused.generator.SOURCES},
        'compile_options_api':api,'compile_options_target':'ascendc_kernels' if api=='npu_op_kernel_options' else 'ALL',
        'only_added_compile_option':'--cce-res-usage','install_or_device_execution':False}


def run(project,output,runner,vendors,merge_project):
    r.require(not output.exists(),'资源报告目录已存在，不覆盖/重启')
    installation=fused.installed(vendors,project,merge_project)
    source_names=fused.generator.SOURCES+('op_kernel/CMakeLists.txt','CMakePresets.json','build.sh')
    before={name:fused.digest(project/name) for name in source_names};runner_sha=fused.digest(runner)
    output.mkdir(parents=True)
    record={'status':'running','accepted_source_sha256':before,'accepted_runner_sha256':runner_sha,
        'scope':'Resource-only compile copy, not installed/executed. No flags or source changes to accepted project/binary.'}
    try:
        # The target machine already confirmed this option in its preceding hardware pass.
        hardware=ROOT/'results/fused_search_256k_q8_close1_hardware.json'
        h=json.loads(hardware.read_text(encoding='utf-8'));compiler=shutil.which('bisheng')
        command=h.get('compiler_help',{}).get('command',[])
        lines=h.get('compiler_help',{}).get('resource_usage_option_lines',[])
        if not (compiler and command and Path(command[0]).resolve()==Path(compiler).resolve()
                and any('--cce-res-usage' in line for line in lines)):
            record.update(status='resource_option_not_confirmed',reason='缺少与当前编译器对应的既有选项证据；不调整环境或猜参数')
        else:
            record['hardware_report_sha256']=fused.digest(hardware)
            record['preparation']=prepare(project,output/'project_copy')
            if record['preparation']['status']!='prepared_resource_only_copy':record['status']='resource_template_unavailable'
            else:
                target=output/'project_copy';log=output/'resource_build.log'
                stage_logs=output/'compiler_logs';stage_logs.mkdir()
                # Official CANN9.2: successful intermediate compiler diagnostics are not persisted by default.
                # Set logging for this child only; never mutate the caller's environment or accepted project.
                env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(stage_logs.resolve())
                record['compiler_logging']={'child_environment':{'ASCENDC_BUILD_LOG_DIR':str(stage_logs.resolve())},
                    'reference':LOGGING_REFERENCE,'parent_environment_modified':False}
                print('仅编译资源报告副本，不安装或执行；实际源文件/128线程/公式保持相同。',flush=True)
                print('编译日志：'+str(log)+'；本步骤不调用msprof或Search程序。',flush=True)
                print('单个编译阶段日志保存至：'+str(stage_logs)+'；只对子进程启用。',flush=True)
                with log.open('wb') as stream:
                    child=subprocess.run(['bash','build.sh'],cwd=target,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
                logs,vf_reports=compiler_reports(output)
                record.update(compiler_log_records=logs,fused_vf_reports=vf_reports)
                r.require(child.returncode==0,'资源副本编译失败，查看 '+str(log)+'；不安装或执行副本')
                options_before=cache(project/'build_out/CMakeCache.txt');options_after=cache(target/'build_out/CMakeCache.txt')
                r.require(options_before and options_before==options_after,'资源副本优化/目标配置不同，不能将其资源数值当作已验收配置')
                for name in fused.generator.SOURCES:r.require(fused.digest(target/name)==before[name],'副本编译修改了计算源码')
                matched=[line for item in logs for line in item.get('resource_context_lines',[])]
                accepted_hashes=set(installation['fused']['device_object_sha256'].values())
                objects={p.relative_to(target).as_posix():fused.digest(p) for p in target.rglob('*.o') if 'op_kernel' in p.parts}
                record.update(status='resource_diagnostic_compile_completed',build_log_sha256=fused.digest(log),
                    configuration=options_after,resource_context_lines=matched,diagnostic_device_objects=objects,
                    any_object_byte_identical_to_accepted=bool(set(objects.values())&accepted_hashes),
                    interpretation='Resource contexts retain function names; absent/unattributed fields remain UNKNOWN. Source/config-matched diagnostic compile is not a newly validated runtime binary.')
                remaining=80
                for item in logs:
                    if item.get('resource_context_lines'):
                        lines=item['resource_context_lines'][:min(40,remaining)]
                        print('资源上下文来源：'+item['path']+'\n'+'\n'.join(lines),flush=True)
                        remaining-=len(lines)
                        if remaining==0:break
                if vf_reports:
                    print('实际融合VF资源报告：'+json.dumps(vf_reports,ensure_ascii=False),flush=True)
                else:
                    print('未找到可归属FusedScoreTopKSimt的完整资源行：寄存器/栈继续记UNKNOWN；不据此判断溢出或自动重编译。',flush=True)
        r.require({name:fused.digest(project/name) for name in source_names}==before and
            fused.digest(runner)==runner_sha and fused.installed(vendors,project,merge_project)==installation,
            '已验收工程/调用器/安装在诊断期间改变')
        record['accepted_artifacts_unchanged']=True;r.write_json(output/'report.json',record)
        print('资源检查状态：'+record['status']+'；已验收产物保持不变。',flush=True)
        return record
    except (Exception,KeyboardInterrupt) as e:
        record.update(status='failed',first_error=str(e));r.write_json(output/'report.json',record);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description='只补融合内核资源编译报告；不重新采集、测速、安装或执行设备程序。')
    defaults={'project':'results/Bm25FusedScoreTopKUbFix1',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'vendors':'results/custom_opp/vendors','merge-project':'results/Bm25TopKMerge'}
    for name,value in defaults.items():parser.add_argument('--'+name,type=Path,default=ROOT/value)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        r.require(sys.platform=='linux','资源副本编译仅在已有CANN的A5 Linux环境执行')
        report=run(args.project.resolve(),args.output.resolve(),args.runner.resolve(),args.vendors.resolve(),args.merge_project.resolve())
        r.require(report['status']=='resource_diagnostic_compile_completed','资源报告未编译完成：'+report['status']+'；查看新目录report.json，首错停止')
    except (Exception,KeyboardInterrupt) as e:
        print('FAILED：'+str(e)+'；不重跑设备采集/性能测试，不安装副本。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
