## Building the docker image

1. Build docker

```
docker image build -t dash-heroku:latest .
```

2. Run docker:

```
docker container run -d -p 6004:6004 dash-heroku
```

App is running at http://localhost:6004/.

## Deploying to Heroku

1. Login to Heroku (first time only):

```
heroku container:login
```

2. Create an app on Heroku (first time only):

```
heroku create <appname>
```

3. Create the container in Heroku:

```
heroku container:push web --app <appname>
```

4. Release the app:

```
heroku container:release web --app <appname>
```
