CONTAINER_NAME=dash-heroku
echo "⛏️ build and run" $CONTAINER_NAME
docker rm -f $CONTAINER_NAME
docker image build -t $CONTAINER_NAME:latest .
docker container run -d -p 6004:6004 --name $CONTAINER_NAME $CONTAINER_NAME 
echo "✅" $CONTAINER_NAME "container is running at http://localhost:6004"