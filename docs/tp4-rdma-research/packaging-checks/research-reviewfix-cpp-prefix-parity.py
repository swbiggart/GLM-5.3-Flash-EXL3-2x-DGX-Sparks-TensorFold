from pathlib import Path
import json,subprocess,tempfile,hashlib,shutil
meta=json.loads(Path('zig/TRANSPORT_SOURCE.json').read_text());path='zig/kernels/cuda/glm_neighbor_rdma.cu';clang=shutil.which('clang++')
with tempfile.TemporaryDirectory(prefix='glm-notice-cpp-') as directory:
    root=Path(directory);(root/'cstdint').touch();(root/'cuda_runtime.h').touch()
    before=subprocess.check_output(['git','-C','/tmp/glm-tp4-rdma-candidates-20261022','show',meta['production']['commit']+':'+path])
    after=(Path('/tmp/glm-tp4-research-reviewfix-source-20261022')/path).read_bytes();outputs=[]
    command=[clang,'-E','-P','-nostdinc','-I',str(root),'-x','c++','-']
    for source in (before,after):outputs.append(subprocess.check_output(command,input=source))
    assert outputs[0]==outputs[1]
    print(json.dumps({'pass':True,'kind':'C++ preprocessor comment-prefix parity','compiler':clang,'command':command,'preprocessed_sha256':hashlib.sha256(outputs[0]).hexdigest(),'limits':['Empty stand-ins for cstdint and cuda_runtime.h used only to isolate preprocessing','No NVCC, CUDA semantic compilation or GPU qualification']},indent=2))
