@transformer
def transform_data(data, *args, **kwargs):
    """
    This block receives data from the loader and prints it.
    'data' is the dictionary returned from the 'load_data' block.
    """
    print("Transformer block is running...")
    print(f"Message received: {data['message']}")

    return {'status': 'success'}