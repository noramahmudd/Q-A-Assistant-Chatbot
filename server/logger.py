import logging

def setup_logger(name="MedicalAssistant"):
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)
    return logger

logger=setup_logger()

logger.info("Logger initialized for Medical Assistant")
logger.debug("Debugging information for Medical Assistant logger")
logger.error("Error message from Medical Assistant logger")
logger.critical("Critical issue in Medical Assistant logger")