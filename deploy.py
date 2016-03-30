import pexpect
import parimoko

submission = pexpect.spawn('scp -r src gpu@rpc.wks.ccs.neu.edu:~/gpudb/');
submission.expect('Password:')
submission.sendline('hf11Ben3ftb7whq')

ssh = parimoko.SSHClient()
ssh.connect('rpc.wks.ccs.neu.edu', username='gpu', password='hf11Ben3ftb7whq')
stdin, stdout, stderr = ssh.exec_command('make check')
data = stdout.read.splitlines();
