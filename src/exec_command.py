import os

def exec_command(exec_cmd, test_flag, logging):
    if test_flag:
        if logging:
            logging.info("Executing: {}".format(exec_cmd))
        else:
            print("Executing: {}".format(exec_cmd))
    else:
        rc = os.system(exec_cmd)
        if rc != 0:
            if logging:
                logging.error("Error running command: {}".format(exec_cmd))
            else:
                print("Error running command: {}".format(exec_cmd))
            exit(1)
