# EBERT: A Knowledge Grounded: Neural Movie Chatbot

![master](https://github.com/rrajasek95/ebert/workflows/Python%20application/badge.svg?branch=master&event=push)

## Local Execution Instructions
Create a `.env` file with the following contents or use the respective environment variables

```
FLASK_APP=bot
FLASK_ENV=development
VERIFY_TOKEN=<VERIFY_TOKEN>
PAGE_ACCESS_TOKEN=<PAGE_ACCESS_TOKEN>
```

The verify token is the token with which you set up the webhook verification process

The page access token is the one for the respective Facebook page which was integrated with the app.

Once that's all done, you should be able to fire off queries to the API!