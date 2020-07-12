import os


class AlpacaPaperSocket(REST):
    def __init__(self):
        if not os.path.exists('../config/creds.yaml')
                key_id_= input('Enter key_id: ')
                secret_key_= input('Enter secret_key: ')
        super().__init__(
            base_url='https://paper-api.alpaca.markets',
            key_id = key_id_,
            secret_key = secret_key_,
        )
