class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


info_dict = {
    'success': {'color': bcolors.OKGREEN, 'comment': 'Success'},
    'fail': {'color': bcolors.FAIL, 'comment': 'Fail'},
    'warning': {'color': bcolors.WARNING, 'comment': 'Warning'},
    'info': {'color': bcolors.HEADER, 'comment': 'Info'}
}


def set_color(status, information):
    status = status.lower()

    return f"{info_dict[status]['color']}[{info_dict[status]['comment']}]{bcolors.ENDC} {information}"