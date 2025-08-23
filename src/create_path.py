import errno
import os

def create_path(topLevel, path, test_flag, logging):
    create_path = os.path.join(topLevel, path)

    if test_flag == False and os.path.exists(path) == False:
        if test_flag:
            logging.info("Test Creating directory: {}".format(create_path))
        else:
            logging.info("Creating directory: {}".format(create_path))
            try:
                os.makedirs(create_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    return create_path
