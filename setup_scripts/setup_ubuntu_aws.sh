#!/bin/sh
#
# Shell script to install nodejs and numpy/scipy python dependencies 
# on clean ubuntu server version >= 12.04 and run SampleSize service.
# with *forever*

# don't forget to open port 80 in your instance's security groups
# see https://forums.aws.amazon.com/thread.jspa?threadID=175685

sudo apt-get install -y git
sudo apt-get install -y python-numpy python-scipy

curl --silent --location https://deb.nodesource.com/setup_0.12 | sudo bash -
sudo apt-get install -y nodejs
sudo npm install -g forever

git clone https://github.com/Anastasia874/SampleSize.git
cd SampleSize
npm install

# TODO: AWS creates nginx by default, so need to place nginx configuration in /etc/nginx.d/
# Currently, Server.js fails with EADDRINUSE error

# Example taken from https://mobiarch.wordpress.com/2014/05/16/creating-an-init-script-in-ubuntu-14-04/
cat > sample_size << EOF
#!/bin/sh

SAMPLE_SIZE_DIR=`pwd`
case "\$1" in
  start)
     sudo EXPRESS_PORT=80 forever start --uid sample_size --append $SAMPLE_SIZE_DIR/Server.js
     ;;
  stop)
     sudo EXPRESS_PORT=80 forever stop sample_size
     ;;
  restart)
     sudo EXPRESS_PORT=80 forever restart sample_size
     ;;
   *)
     echo "Usage: ./server.sh {start|stop|restart}" >&2
     exit 3
;;
esac
EOF

chmod a+x sample_size
sudo cp sample_size /etc/init.d/

# Register the script as an init script:
sudo update-rc.d sample_size defaults