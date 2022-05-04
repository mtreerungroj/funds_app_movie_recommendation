# Movie Recommendation

This project aims to study about movie recommendation systems using [BERT](https://github.com/google-research/bert) and was deployed on [Heroku](https://www.heroku.com/).

🚀✨ [Demo](https://fundsappwithdocker.herokuapp.com/) ✨🚀

## Building the docker image

```
. ./run.sh
```

App is running on http://localhost:6006/.

## Deploying to Heroku

1. Login to Heroku and create an app (first time only):

```
heroku container:login
heroku create <appname>
```

2. Deploy:

```
. ./deploy.sh
```
