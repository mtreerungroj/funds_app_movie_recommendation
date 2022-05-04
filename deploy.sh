APPNAME=fundsappwithdocker
echo "🚀 deploy to" $APPNAME
heroku container:push web --app $APPNAME
heroku container:release web --app $APPNAME
APPNAME=fundsappwithdocker
echo "✅ deployed. 👉 https://$APPNAME.herokuapp.com/"