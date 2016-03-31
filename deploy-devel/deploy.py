import pexpect
import paramiko
import os

def runMakeCheck():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname='rpc.wks.ccs.neu.edu', username='gpu',
        password='hf11Ben3ftb7whq')
    except Exception as e:
        print e;

    channel = ssh.get_transport().open_session()
    channel.exec_command('source ~/.profile && cd ~/gpudb/gpu-no-sql/src && make check')
    channel.shutdown_write()

    stdout = channel.makefile().read()
    stderr = channel.makefile_stderr().read()
    exit_code = channel.recv_exit_status()

    channel.close()
    ssh.close()

    print('stdout:\n' + stdout)
    print('stderr:' + stderr)
    print('exit_code:' + str(exit_code))

try:
    cwd = os.path.dirname(os.path.realpath(__file__))
    spawnCommand = 'scp -r ' + cwd + ' gpu@rpc.wks.ccs.neu.edu:~/gpudb/'
    print spawnCommand
    submission = pexpect.spawn(command = spawnCommand)

    submissionResult = submission.expect(["Password:", pexpect.EOF])
    if submissionResult is not 0:
        print "Connection timeout"
        exit(1);
    else:
        submission.sendline('hf11Ben3ftb7whq')
        submission.expect(pexpect.EOF)
        print('copy to remote complete, running make check')
        runMakeCheck()

except Exception as e:
    print e;


