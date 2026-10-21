"""Resource-only build of a verified copy. Never install/run its output or modify the accepted project."""
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
    pattern=re.compile(r'FusedScoreTopKSimt|Stack[_\s]*size|register(?:s|\s+count|\s+size|\s+usage)|\b(?:GPR|VGPR|SGPR|RegNum|UsedRegs)\b|spill',re.I)
    for i,line in enumerate(lines):
        if pattern.search(line):selected.update(range(max(0,i-2),min(len(lines),i+4)))
    return [lines[i][:1000] for i in sorted(selected)][:120]


def prepare(project,target):
    """Require the actual generated Debug hook signature before adding its known OPTIONS argument."""
    project=project.resolve(strict=True);target=target.resolve()
    r.require(not target.exists() and target.is_relative_to((ROOT/'results').resolve()) and
        target!=project and not target.is_relative_to(project) and not project.is_relative_to(target),
        '资源副本必须是results下新独立目录，不能覆盖/嵌套原工程')
    cmake=project/'op_kernel/CMakeLists.txt';text=cmake.read_text(encoding='utf-8')
    # Same hook used by group IndexSDK; match the actual generated file rather than assume it exists.
    hook=re.search(r'(?im)^[ \t]*if\s*\([^\n]*CMAKE_BUILD_TYPE[^\n]*\)\s*\n[ \t]*add_ops_compile_options\s*\(\s*ALL\s+OPTIONS\b',text)
    if not hook:return {'status':'unsupported_actual_template','actual_cmake_sha256':fused.digest(cmake),
        'reason':'实际模板未提供已核对的Debug add_ops_compile_options(ALL OPTIONS ...)钩子；不猜新API，不编译'}
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
    added='\n# Resource diagnostic COPY ONLY; no precision/optimization/thread changes.\nadd_ops_compile_options(ALL OPTIONS --cce-res-usage)\n'
    (target/'op_kernel/CMakeLists.txt').write_bytes((text[:hook.start()]+added+text[hook.start():]).encode('utf-8'))
    return {'status':'prepared_resource_only_copy','actual_cmake_sha256':fused.digest(cmake),
        'copy_cmake_sha256':fused.digest(target/'op_kernel/CMakeLists.txt'),
        'source_sha256':{name:fused.digest(target/name) for name in fused.generator.SOURCES},
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
                print('仅编译资源报告副本，不安装或执行；实际源文件/128线程/公式保持相同。',flush=True)
                with log.open('wb') as stream:
                    child=subprocess.run(['bash','build.sh'],cwd=target,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
                r.require(child.returncode==0,'资源副本编译失败，查看 '+str(log)+'；不安装或执行副本')
                options_before=cache(project/'build_out/CMakeCache.txt');options_after=cache(target/'build_out/CMakeCache.txt')
                r.require(options_before and options_before==options_after,'资源副本优化/目标配置不同，不能将其资源数值当作已验收配置')
                for name in fused.generator.SOURCES:r.require(fused.digest(target/name)==before[name],'副本编译修改了计算源码')
                text=log.read_text(encoding='utf-8',errors='replace');matched=resource_lines(text)
                accepted_hashes=set(installation['fused']['device_object_sha256'].values())
                objects={p.relative_to(target).as_posix():fused.digest(p) for p in target.rglob('*.o') if 'op_kernel' in p.parts}
                record.update(status='resource_diagnostic_compile_completed',build_log_sha256=fused.digest(log),
                    configuration=options_after,resource_context_lines=matched,diagnostic_device_objects=objects,
                    any_object_byte_identical_to_accepted=bool(set(objects.values())&accepted_hashes),
                    interpretation='Resource contexts retain function names; absent/unattributed fields remain UNKNOWN. Source/config-matched diagnostic compile is not a newly validated runtime binary.')
                print('寄存器/栈资源上下文：\n'+'\n'.join(matched[:65]) if matched else '编译未输出可识别的VF寄存器/栈字段；仍记UNKNOWN。',flush=True)
        r.require({name:fused.digest(project/name) for name in source_names}==before and
            fused.digest(runner)==runner_sha and fused.installed(vendors,project,merge_project)==installation,
            '已验收工程/调用器/安装在诊断期间改变')
        record['accepted_artifacts_unchanged']=True;r.write_json(output/'report.json',record)
        print('资源检查状态：'+record['status']+'；已验收产物保持不变。',flush=True)
        return record
    except (Exception,KeyboardInterrupt) as e:
        record.update(status='failed',first_error=str(e));r.write_json(output/'report.json',record);raise
