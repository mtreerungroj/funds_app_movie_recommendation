CONTAINER_NAME=dash-heroku
echo "⛏️ build and run" $CONTAINER_NAME
docker rm -f $CONTAINER_NAME
docker image build -t ${CONTAINER_NAME}:latest .
docker container run --memory="1g" -it -d -p 6005:6005 --name $CONTAINER_NAME $CONTAINER_NAME 
echo "✅" $CONTAINER_NAME "container is running at http://localhost:6005"