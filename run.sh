CONTAINER_NAME=dash-heroku
echo "⛏️ build and run" $CONTAINER_NAME
docker rm -f $CONTAINER_NAME
docker image build --no-cache -t ${CONTAINER_NAME}:latest .
docker container run -it -d -p 6006:6006 --name $CONTAINER_NAME $CONTAINER_NAME 
echo "✅" $CONTAINER_NAME "container is running at http://localhost:6006"