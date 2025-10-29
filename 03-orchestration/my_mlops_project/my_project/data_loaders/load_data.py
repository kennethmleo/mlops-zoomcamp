@data_loader
def load_data_from_sources(*args, **kwargs):
    """
    This is our first "hello" message.
    """
    message = "Hello from the Data Loader!"
    print(message)

    # Data loaders must return a dictionary
    return {'message': message}