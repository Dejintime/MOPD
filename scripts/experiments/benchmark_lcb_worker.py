"""Runs ONLY inside the isolated bubblewrap namespace, never on the host."""
import contextlib, io, json, resource, sys
payload=json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,)*2)
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
resource.setrlimit(resource.RLIMIT_CPU,(payload['cpu_limit'],)*2)
import numpy as np
from testing_util import run_test
output=sys.stdout
with open('/tmp/judge-stdout.log','w') as log_out, open('/tmp/judge-stderr.log','w') as log_err, contextlib.redirect_stdout(log_out), contextlib.redirect_stderr(log_err):
    res, metadata=run_test(payload['sample'],test=payload['code'],debug=False,timeout=6)
res=[x.item() if isinstance(x,np.generic) else x for x in res]
output.write(json.dumps({'tests':res,'metadata':metadata})+'\n');output.flush()
