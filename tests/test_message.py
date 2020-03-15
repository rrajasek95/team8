from bot.message import generate_echo_message

def test_generate_echo_message():
    """
    Naive test to check if message is in the right format to
    send to the FB message API
    """
    response = generate_echo_message("1", "TEST")
    assert response["recipient"]["id"] == "1"
    assert response["message"]["text"] == "TEST" 