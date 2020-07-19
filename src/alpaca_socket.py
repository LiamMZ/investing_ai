import os
import yaml
import alpaca_trade_api as tradeapi

class AlpacaPaperSocket(tradeapi.REST):
    def __init__(self):
        if os.path.exists('../config/creds.yaml'):
            with open('../config/creds.yaml') as file:
                creds = yaml.load(file, Loader=yaml.FullLoader)
            key_id_ = creds['key_id']
            secret_key_ = creds['secret_key']   
        else:
            key_id_= input('Enter key_id: ')
            secret_key_= input('Enter secret_key: ')

        super().__init__(
            base_url='https://paper-api.alpaca.markets',
            key_id = key_id_,
            secret_key = secret_key_,
        )
