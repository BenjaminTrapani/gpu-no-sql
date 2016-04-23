import pexpect
import paramiko
import os

def runSSHWithCommand(command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname='rpc.wks.ccs.neu.edu', username='gpu',
        password='hf11Ben3ftb7whq')
    except Exception as e:
        print e;

    channel = ssh.get_transport().open_session()
    channel.exec_command(command = command)
    channel.shutdown_write()

    stdout = channel.makefile().read()
    stderr = channel.makefile_stderr().read()
    exit_code = channel.recv_exit_status()

    channel.close()
    ssh.close()

    print('stdout:\n' + stdout)
    print('stderr:' + stderr)
    print('exit_code:' + str(exit_code))

def runMakeCheck():
    runSSHWithCommand('source ~/.profile && cd ~/gpudb/gpu-no-sql/src && make clean &&  make check')

def runMakeTopCheck():
    runSSHWithCommand('source ~/.profile && cd ~/gpudb/gpu-no-sql/src && make clean &&  make topcheck')

def cleanOldGPUDB():
    print('removing old files in ~/gpudb/gpu-no-sql/')
    runSSHWithCommand('rm -rf ~/gpudb/gpu-no-sql/')

try:
    cleanOldGPUDB()
    cwd = os.path.dirname(os.path.realpath(__file__))
    spawnCommand = 'scp -r ' + cwd + ' gpu@rpc.wks.ccs.neu.edu:~/gpudb/'
    print spawnCommand
    submission = pexpect.spawn(command = spawnCommand, timeout=300)
    submission.timeout = 300
    submissionResult = submission.expect(["Password:", pexpect.EOF])
    if submissionResult is not 0:
        print "Connection timeout"
        exit(1);
    else:
        submission.sendline('hf11Ben3ftb7whq')
        submission.expect(pexpect.EOF)
        #print('copy to remote complete, running make check')
        #runMakeCheck()
        print('copy to remote complete, running make topcheck')
        runMakeTopCheck()

except Exception as e:
    print e;


