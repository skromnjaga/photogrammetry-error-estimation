# Путь к файлам проекта для управления тестовой установкой
IPCT_TEST_SETUP_PATH = r"D:\ipct_test_setup\IPCT_test_setup" 
# IPCT_TEST_SETUP_PATH = r"C:\Users\Caesar\source\repos\ipct_test_setup\IPCT_test_setup"

# Путь к файлам проекта для проведения фазограмметрических измерений
# STRUCTURE_LIGHT_PYTHON_PATH = r"C:\Users\User\source\repos\helleb0re\structured-light-python"
STRUCTURE_LIGHT_PYTHON_PATH = r"C:\Users\Caesar\source\repos\helleb0re\structured-light-python"

# Папка для сохранения результатов
RESULTS_PATH = "experimental_results"

# Имена портов для подключения к сервомашинам и лазерному датчику
SERVO_PORT = "COM3"
SENSOR_PORT = "COM4"

# Тип камер стереосистемы и фазограмметрической системы
STEREO_CAMS_TYPE = 'baumer'
PHASE_CAMS_TYPE = 'baumer'

# Серийные номера камер стереосистемы и фазограмметрической системы
STEREO_CAMS_SERIAL_NUMBERS = ['700005466452', '700005466457']
PHASE_CAMS_SERIAL_NUMBERS = ['700007464735', '700007464734']

STEREO_CAMS_EXPOSURES = [20_000, 20_000] # us
STEREO_CAMS_GAINS = [1.0, 1.0]

PHASE_CAMS_EXPOSURES = [20_000, 20_000] # us
PHASE_CAMS_GAINS = [1.0, 1.0]