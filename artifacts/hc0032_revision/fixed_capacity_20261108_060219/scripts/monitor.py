import sys,os,time,json,subprocess,datetime
pid=int(sys.argv[1])
with open(sys.argv[2],'x') as f:
 while True:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  s=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  free,used,util=[int(x.strip()) for x in s.stdout.split(',')]
  f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'free_mib':free,'used_mib':used,'utilization':util})+'\n');f.flush();time.sleep(.5)
