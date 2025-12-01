from datetime import datetime


def isonow():
    return datetime.now().replace(microsecond=0).isoformat()
