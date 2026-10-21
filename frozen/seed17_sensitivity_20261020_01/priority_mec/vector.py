"""Synchronous CPU environments, stable episode IDs and complete worker state."""
import multiprocessing as mp
import traceback
import numpy as np

def _worker(conn,cfg,seed,control):
    from .environment import UAVMECEnv
    env=UAVMECEnv(cfg,seed=seed,control=control)
    while True:
        try:
            command,arg=conn.recv()
            if command=='close':conn.send(('ok',None));break
            if command=='reset':result=env.reset(episode=arg)
            elif command=='step':result=env.step(arg)
            elif command=='prepare':result=env.prepare_service()
            elif command=='state':result=env.state_dict()
            elif command=='load':result=env.load_state_dict(arg)
            elif command=='sizes':result=(env.observation_size,env.action_size)
            else:raise ValueError(command)
            conn.send(('ok',result))
        except EOFError:break
        except BaseException:conn.send(('error',traceback.format_exc()))
    conn.close()

class VectorEnvs:
    def __init__(self,cfg,seed,control,count,processes=True):
        self.count=count;self.remote=bool(processes and count>1);self.workers=[];self.pipes=[]
        if self.remote:
            ctx=mp.get_context('spawn')
            for _ in range(count):
                parent,child=ctx.Pipe();worker=ctx.Process(target=_worker,args=(child,cfg,seed,control),daemon=True)
                worker.start();child.close();self.pipes.append(parent);self.workers.append(worker)
            self.pipes[0].send(('sizes',None));self.observation_size,self.action_size=self._recv(self.pipes[0])
        else:
            from .environment import UAVMECEnv
            self.envs=[UAVMECEnv(cfg,seed=seed,control=control) for _ in range(count)]
            self.observation_size=self.envs[0].observation_size;self.action_size=self.envs[0].action_size
    @staticmethod
    def _recv(pipe):
        state,value=pipe.recv()
        if state!='ok':raise RuntimeError('Environment worker failed:\n'+value)
        return value
    def all(self,command,args):
        if len(args)!=self.count:raise ValueError('Exactly one argument per environment required')
        if self.remote:
            for pipe,arg in zip(self.pipes,args):pipe.send((command,arg))
            return [self._recv(pipe) for pipe in self.pipes]
        if command=='reset':return [e.reset(episode=a) for e,a in zip(self.envs,args)]
        if command=='step':return [e.step(a) for e,a in zip(self.envs,args)]
        if command=='prepare':return [e.prepare_service() for e in self.envs]
        if command=='state':return [e.state_dict() for e in self.envs]
        if command=='load':return [e.load_state_dict(a) for e,a in zip(self.envs,args)]
        raise ValueError(command)
    def reset_one(self,index,episode):
        if self.remote:
            self.pipes[index].send(('reset',episode));return self._recv(self.pipes[index])
        return self.envs[index].reset(episode=episode)
    def prepare_one(self,index):
        if self.remote:
            self.pipes[index].send(('prepare',None));return self._recv(self.pipes[index])
        return self.envs[index].prepare_service()
    def close(self):
        if self.remote:
            for pipe in self.pipes:
                try:pipe.send(('close',None))
                except (BrokenPipeError,EOFError,OSError):pass
            for worker in self.workers:
                worker.join(timeout=3)
                if worker.is_alive():worker.terminate();worker.join(timeout=3)
            for pipe in self.pipes:pipe.close()
