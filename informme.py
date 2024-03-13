
import requests
import sys

experiment_name=sys.argv[1]
#push message to url with AL and budget as title
requests.get(f'https://wirepusher.com/send?id=hbBompXx6&title={experiment_name}&message=completed')