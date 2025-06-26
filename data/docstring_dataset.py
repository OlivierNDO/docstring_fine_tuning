# File: docstring_dataset.py

DOCSTRING_EXAMPLES = [
{
    "instruction": "def send_email(to_address, subject, body):\n    smtp.send(to_address, subject, body)",
    "response": '"""Send an email with a subject and body.\n\nArgs:\n    to_address: Recipient email address\n    subject: Subject of the email\n    body: Content of the email\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def load_json_file(path):\n    with open(path, 'r') as f:\n        return json.load(f)",
    "response": '"""Load JSON data from a file.\n\nArgs:\n    path: Path to the JSON file\n\nReturns:\n    Parsed JSON content as a dictionary\n"""'
},
{
    "instruction": "def calculate_discount(price, percent):\n    return price * (1 - percent / 100)",
    "response": '"""Calculate discounted price.\n\nArgs:\n    price: Original price\n    percent: Discount percentage\n\nReturns:\n    Price after discount\n"""'
},
{
    "instruction": "def parse_date(date_str):\n    return datetime.strptime(date_str, '%Y-%m-%d')",
    "response": '"""Convert a string to a datetime object.\n\nArgs:\n    date_str: Date string in YYYY-MM-DD format\n\nReturns:\n    datetime object\n"""'
},
{
    "instruction": "def get_file_extension(filename):\n    return filename.split('.')[-1]",
    "response": '"""Extract the file extension from a filename.\n\nArgs:\n    filename: Name of the file\n\nReturns:\n    File extension as string\n"""'
},
{
    "instruction": "class Config:\n    def __init__(self, config_path):\n        self.settings = json.load(open(config_path))",
    "response": '"""Loads configuration from a JSON file.\n\nAttributes:\n    settings: Dictionary of configuration parameters\n"""'
},
{
    "instruction": "def validate_email(email):\n    return re.match(r\"[^@]+@[^@]+\\.[^@]+\", email) is not None",
    "response": '"""Validate an email address format.\n\nArgs:\n    email: Email string to validate\n\nReturns:\n    True if valid, else False\n"""'
},
{
    "instruction": "def read_csv(path):\n    return pd.read_csv(path)",
    "response": '"""Read a CSV file into a DataFrame.\n\nArgs:\n    path: Path to the CSV file\n\nReturns:\n    pandas DataFrame\n"""'
},
{
    "instruction": "def count_words(text):\n    return len(text.split())",
    "response": '"""Count the number of words in a string.\n\nArgs:\n    text: Input text\n\nReturns:\n    Integer word count\n"""'
},
{
    "instruction": "def normalize_vector(vec):\n    norm = np.linalg.norm(vec)\n    return vec / norm if norm != 0 else vec",
    "response": '"""Normalize a vector to unit length.\n\nArgs:\n    vec: Numpy array representing a vector\n\nReturns:\n    Normalized vector\n"""'
},
{
    "instruction": "class Timer:\n    def __init__(self):\n        self.start_time = time.time()",
    "response": '"""Simple timer utility.\n\nAttributes:\n    start_time: Timestamp when the timer was initialized\n"""'
},
{
    "instruction": "def get_top_k(items, k):\n    return sorted(items, reverse=True)[:k]",
    "response": '"""Get the top-k highest values from a list.\n\nArgs:\n    items: List of values\n    k: Number of top values to return\n\nReturns:\n    List of top-k values\n"""'
},
{
    "instruction": "def format_currency(amount):\n    return '${:,.2f}'.format(amount)",
    "response": '"""Format a number as a US dollar currency string.\n\nArgs:\n    amount: Numeric value to format\n\nReturns:\n    Formatted currency string\n"""'
},
{
    "instruction": "def extract_domain(url):\n    return url.split('//')[-1].split('/')[0]",
    "response": '"""Extract domain from a URL.\n\nArgs:\n    url: URL string\n\nReturns:\n    Domain part of the URL\n"""'
},
{
    "instruction": "def remove_duplicates(seq):\n    return list(set(seq))",
    "response": '"""Remove duplicate elements from a list.\n\nArgs:\n    seq: List of elements\n\nReturns:\n    List with duplicates removed\n"""'
},
{
    "instruction": "def write_to_file(path, text):\n    with open(path, 'w') as f:\n        f.write(text)",
    "response": '"""Write text to a file.\n\nArgs:\n    path: File path\n    text: Text content to write\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def chunk_list(lst, size):\n    return [lst[i:i+size] for i in range(0, len(lst), size)]",
    "response": '"""Split a list into chunks of a given size.\n\nArgs:\n    lst: List to split\n    size: Chunk size\n\nReturns:\n    List of list chunks\n"""'
},
{
    "instruction": "def get_env_variable(key, default=None):\n    return os.environ.get(key, default)",
    "response": '"""Fetch an environment variable with fallback.\n\nArgs:\n    key: Environment variable name\n    default: Default value if key is not found\n\nReturns:\n    Value of the environment variable or default\n"""'
},
{
    "instruction": "def send_sms(phone_number, message):\n    sms_gateway.send(phone_number, message)",
    "response": '"""Send an SMS message.\n\nArgs:\n    phone_number: Recipient phone number\n    message: Text content to send\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def get_weekday(date_str):\n    return datetime.strptime(date_str, '%Y-%m-%d').strftime('%A')",
    "response": '"""Get the weekday name for a given date.\n\nArgs:\n    date_str: Date in YYYY-MM-DD format\n\nReturns:\n    Name of the weekday\n"""'
},
{
    "instruction": "def remove_html_tags(text):\n    return re.sub(r'<[^>]+>', '', text)",
    "response": '"""Remove HTML tags from a string.\n\nArgs:\n    text: String containing HTML\n\nReturns:\n    Cleaned text without tags\n"""'
},
{
    "instruction": "def calculate_bmi(weight, height):\n    return weight / (height ** 2)",
    "response": '"""Calculate Body Mass Index (BMI).\n\nArgs:\n    weight: Weight in kilograms\n    height: Height in meters\n\nReturns:\n    BMI as float\n"""'
},
{
    "instruction": "def save_pickle(obj, path):\n    with open(path, 'wb') as f:\n        pickle.dump(obj, f)",
    "response": '"""Serialize an object and save it to disk.\n\nArgs:\n    obj: Object to serialize\n    path: File path to save the object\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def count_occurrences(items, target):\n    return items.count(target)",
    "response": '"""Count the number of times an item appears in a list.\n\nArgs:\n    items: List of elements\n    target: Element to count\n\nReturns:\n    Integer count\n"""'
},
{
    "instruction": "def capitalize_words(sentence):\n    return ' '.join(word.capitalize() for word in sentence.split())",
    "response": '"""Capitalize the first letter of each word in a sentence.\n\nArgs:\n    sentence: Input string\n\nReturns:\n    Capitalized sentence\n"""'
},
{
    "instruction": "def zip_lists(keys, values):\n    return dict(zip(keys, values))",
    "response": '"""Combine two lists into a dictionary.\n\nArgs:\n    keys: List of keys\n    values: List of values\n\nReturns:\n    Dictionary mapping keys to values\n"""'
},
{
    "instruction": "def wait(seconds):\n    time.sleep(seconds)",
    "response": '"""Pause execution for a specified duration.\n\nArgs:\n    seconds: Number of seconds to wait\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def clear_cache():\n    cache.clear()",
    "response": '"""Clear the in-memory cache.\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def truncate_text(text, max_length):\n    return text[:max_length] + '...' if len(text) > max_length else text",
    "response": '"""Truncate text to a maximum length.\n\nArgs:\n    text: Input string\n    max_length: Maximum allowed length\n\nReturns:\n    Truncated string\n"""'
},
{
    "instruction": "def find_max_index(numbers):\n    return numbers.index(max(numbers))",
    "response": '"""Find the index of the maximum value in a list.\n\nArgs:\n    numbers: List of numbers\n\nReturns:\n    Index of the largest number\n"""'
},
{
    "instruction": "def has_duplicates(seq):\n    return len(seq) != len(set(seq))",
    "response": '"""Check if a list contains duplicate elements.\n\nArgs:\n    seq: List of elements\n\nReturns:\n    True if duplicates exist, else False\n"""'
},
{
    "instruction": "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9",
    "response": '"""Convert temperature from Fahrenheit to Celsius.\n\nArgs:\n    f: Temperature in Fahrenheit\n\nReturns:\n    Temperature in Celsius\n"""'
},
{
    "instruction": "def get_last_element(lst):\n    return lst[-1] if lst else None",
    "response": '"""Return the last element in a list.\n\nArgs:\n    lst: Input list\n\nReturns:\n    Last item or None if list is empty\n"""'
},
{
    "instruction": "def pad_string(s, length):\n    return s.ljust(length)",
    "response": '"""Pad a string with spaces to the right.\n\nArgs:\n    s: Input string\n    length: Desired total length\n\nReturns:\n    Right-padded string\n"""'
},
{
    "instruction": "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height",
    "response": '"""Represents a rectangle shape.\n\nAttributes:\n    width: Width of the rectangle\n    height: Height of the rectangle\n"""'
},
{
    "instruction": "def is_palindrome(word):\n    return word == word[::-1]",
    "response": '"""Check if a word is a palindrome.\n\nArgs:\n    word: String to evaluate\n\nReturns:\n    True if palindrome, else False\n"""'
},
{
    "instruction": "def extract_digits(s):\n    return ''.join(filter(str.isdigit, s))",
    "response": '"""Extract numeric digits from a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    String containing only digits\n"""'
},
{
    "instruction": "def list_diff(a, b):\n    return list(set(a) - set(b))",
    "response": '"""Compute the difference between two lists.\n\nArgs:\n    a: First list\n    b: Second list\n\nReturns:\n    List of elements in a but not in b\n"""'
},
{
    "instruction": "def generate_slug(text):\n    return text.lower().replace(' ', '-')",
    "response": '"""Generate a URL-friendly slug from text.\n\nArgs:\n    text: Input string\n\nReturns:\n    Slugified version of the string\n"""'
},
{
    "instruction": "def get_unique_elements(seq):\n    return list(dict.fromkeys(seq))",
    "response": '"""Remove duplicates while preserving order.\n\nArgs:\n    seq: List of elements\n\nReturns:\n    List of unique elements\n"""'
},
{
    "instruction": "def hex_to_rgb(hex_color):\n    hex_color = hex_color.lstrip('#')\n    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))",
    "response": '"""Convert hex color code to RGB tuple.\n\nArgs:\n    hex_color: Color in hex format (e.g., '#ff00ff')\n\nReturns:\n    Tuple of RGB values\n"""'
},
{
    "instruction": "def filter_positive(numbers):\n    return [n for n in numbers if n > 0]",
    "response": '"""Filter out non-positive numbers.\n\nArgs:\n    numbers: List of integers or floats\n\nReturns:\n    List of positive numbers\n"""'
},
{
    "instruction": "def format_iso_date(dt):\n    return dt.strftime('%Y-%m-%dT%H:%M:%S')",
    "response": '"""Format a datetime object to ISO 8601.\n\nArgs:\n    dt: datetime object\n\nReturns:\n    ISO 8601 formatted string\n"""'
},
{
    "instruction": "def get_middle_index(items):\n    return len(items) // 2",
    "response": '"""Get the index of the middle element.\n\nArgs:\n    items: List of elements\n\nReturns:\n    Index of the middle element\n"""'
},
{
    "instruction": "def parse_bool(value):\n    return str(value).lower() in ['true', '1', 'yes']",
    "response": '"""Parse a string as a boolean value.\n\nArgs:\n    value: String or other input\n\nReturns:\n    Boolean interpretation\n"""'
},
{
    "instruction": "def get_file_size(path):\n    return os.path.getsize(path)",
    "response": '"""Get the size of a file in bytes.\n\nArgs:\n    path: File path\n\nReturns:\n    File size in bytes\n"""'
},
{
    "instruction": "def get_extension(path):\n    return os.path.splitext(path)[1]",
    "response": '"""Extract the file extension.\n\nArgs:\n    path: Path to the file\n\nReturns:\n    File extension as string\n"""'
},
{
    "instruction": "def flatten_list(nested):\n    return [item for sublist in nested for item in sublist]",
    "response": '"""Flatten a list of lists.\n\nArgs:\n    nested: List containing sublists\n\nReturns:\n    Flattened list\n"""'
},
{
    "instruction": "def contains_only_digits(s):\n    return s.isdigit()",
    "response": '"""Check if a string contains only digits.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if all characters are digits\n"""'
},
{
    "instruction": "def get_first_non_none(*args):\n    for arg in args:\n        if arg is not None:\n            return arg",
    "response": '"""Return the first non-None argument.\n\nArgs:\n    *args: Variable length argument list\n\nReturns:\n    First non-None value or None\n"""'
},
{
    "instruction": "def get_initials(name):\n    return ''.join(part[0].upper() for part in name.split())",
    "response": '"""Get initials from a full name.\n\nArgs:\n    name: Full name string\n\nReturns:\n    Initials as uppercase string\n"""'
},
{
    "instruction": "def round_up(n, decimals=0):\n    multiplier = 10 ** decimals\n    return math.ceil(n * multiplier) / multiplier",
    "response": '"""Round a number up to a specified number of decimals.\n\nArgs:\n    n: Number to round\n    decimals: Number of decimal places\n\nReturns:\n    Rounded value\n"""'
},
{
    "instruction": "def reverse_dict(d):\n    return {v: k for k, v in d.items()}",
    "response": '"""Reverse keys and values in a dictionary.\n\nArgs:\n    d: Dictionary to reverse\n\nReturns:\n    Dictionary with keys and values swapped\n"""'
},
{
    "instruction": "def safe_divide(a, b):\n    return a / b if b != 0 else None",
    "response": '"""Safely divide two numbers.\n\nArgs:\n    a: Numerator\n    b: Denominator\n\nReturns:\n    Result or None if division by zero\n"""'
},
{
    "instruction": "def camel_to_snake(name):\n    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()",
    "response": '"""Convert CamelCase to snake_case.\n\nArgs:\n    name: CamelCase string\n\nReturns:\n    snake_case version of the string\n"""'
},
{
    "instruction": "def rotate_list(lst, n):\n    return lst[n:] + lst[:n]",
    "response": '"""Rotate a list by n positions.\n\nArgs:\n    lst: List of elements\n    n: Number of positions to rotate\n\nReturns:\n    Rotated list\n"""'
},
{
    "instruction": "def repeat_string(s, times):\n    return s * times",
    "response": '"""Repeat a string multiple times.\n\nArgs:\n    s: String to repeat\n    times: Number of repetitions\n\nReturns:\n    Repeated string\n"""'
},
{
    "instruction": "def get_class_name(obj):\n    return obj.__class__.__name__",
    "response": '"""Get the class name of an object.\n\nArgs:\n    obj: Instance of a class\n\nReturns:\n    Class name as string\n"""'
},
{
    "instruction": "Add two numbers and return the result.",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "SELECT * FROM users WHERE active = 1;",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "<div class='header'>Welcome</div>",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "CREATE TABLE customers (id INT, name TEXT);",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "echo 'Hello, world!'",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "for (int i = 0; i < 10; i++) {\n    printf(\"%d\", i);\n}",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "#!/bin/bash\nls -la",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "function greet(name) {\n    return 'Hello ' + name;\n}",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "1. Heat oven to 350°F.\n2. Mix ingredients.\n3. Bake for 25 minutes.",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "name: Build and Test\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "Here’s a paragraph explaining how ecosystems work...",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "import os and check the environment variable",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "definitely not code at all",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "Run a regression using sklearn and plot the results",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "**Bold Text** and _Italic Text_",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "To be or not to be, that is the question.",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "classy and functional but no actual code",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "This should explain how to implement caching in Redis.",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "Translate this sentence to Spanish: 'Good morning'",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
    "instruction": "Remind me to call John at 3pm tomorrow.",
    "response": "Sorry, this doesn't look like a python function or class to me."
},
{
"instruction": "def merge_dicts(dict1, dict2):\n    result = dict1.copy()\n    result.update(dict2)\n    return result",
"response": '"""Merge two dictionaries into one.\n\nArgs:\n    dict1: First dictionary\n    dict2: Second dictionary\n\nReturns:\n    Merged dictionary with dict2 values overriding dict1\n"""'
},
{
    "instruction": "def get_random_color():\n    import random\n    return '#{:06x}'.format(random.randint(0, 0xFFFFFF))",
    "response": '"""Generate a random hex color code.\n\nReturns:\n    Random color in hex format (e.g., '#a1b2c3')\n"""'
},
{
    "instruction": "def clamp(value, min_val, max_val):\n    return max(min_val, min(value, max_val))",
    "response": '"""Constrain a value within a range.\n\nArgs:\n    value: Value to clamp\n    min_val: Minimum allowed value\n    max_val: Maximum allowed value\n\nReturns:\n    Clamped value within the specified range\n"""'
},
{
    "instruction": "def check_password_strength(password):\n    import re\n    return len(password) >= 8 and re.search(r'[A-Z]', password) and re.search(r'[0-9]', password)",
    "response": '"""Check if a password meets basic strength requirements.\n\nArgs:\n    password: Password string to validate\n\nReturns:\n    True if password is strong, else False\n"""'
},
{
    "instruction": "def create_backup(source, destination):\n    import shutil\n    shutil.copy2(source, destination)",
    "response": '"""Create a backup copy of a file.\n\nArgs:\n    source: Path to the source file\n    destination: Path for the backup file\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def generate_uuid():\n    import uuid\n    return str(uuid.uuid4())",
    "response": '"""Generate a unique identifier.\n\nReturns:\n    UUID string\n"""'
},
{
    "instruction": "def sort_by_length(strings):\n    return sorted(strings, key=len)",
    "response": '"""Sort a list of strings by their length.\n\nArgs:\n    strings: List of strings\n\nReturns:\n    List sorted by string length\n"""'
},
{
    "instruction": "def get_weekdays(start_date, end_date):\n    import datetime\n    days = []\n    current = start_date\n    while current <= end_date:\n        if current.weekday() < 5:\n            days.append(current)\n        current += datetime.timedelta(days=1)\n    return days",
    "response": '"""Get all weekdays between two dates.\n\nArgs:\n    start_date: Starting date\n    end_date: Ending date\n\nReturns:\n    List of datetime objects for weekdays only\n"""'
},
{
    "instruction": "def convert_to_title_case(text):\n    return text.title()",
    "response": '"""Convert text to title case.\n\nArgs:\n    text: Input string\n\nReturns:\n    String with first letter of each word capitalized\n"""'
},
{
    "instruction": "def calculate_age(birth_date):\n    import datetime\n    today = datetime.date.today()\n    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))",
    "response": '"""Calculate age in years from birth date.\n\nArgs:\n    birth_date: Date of birth\n\nReturns:\n    Age in years as integer\n"""'
},
{
    "instruction": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "response": '"""Check if a number is prime.\n\nArgs:\n    n: Integer to check\n\nReturns:\n    True if number is prime, else False\n"""'
},
{
    "instruction": "def encode_base64(text):\n    import base64\n    return base64.b64encode(text.encode()).decode()",
    "response": '"""Encode text to base64.\n\nArgs:\n    text: Text to encode\n\nReturns:\n    Base64 encoded string\n"""'
},
{
    "instruction": "def get_file_hash(filepath):\n    import hashlib\n    with open(filepath, 'rb') as f:\n        return hashlib.md5(f.read()).hexdigest()",
    "response": '"""Calculate MD5 hash of a file.\n\nArgs:\n    filepath: Path to the file\n\nReturns:\n    MD5 hash as hexadecimal string\n"""'
},
{
    "instruction": "def retry_operation(func, max_attempts=3):\n    for attempt in range(max_attempts):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_attempts - 1:\n                raise e\n            time.sleep(1)",
    "response": '"""Retry a function with exponential backoff.\n\nArgs:\n    func: Function to retry\n    max_attempts: Maximum number of attempts\n\nReturns:\n    Result of successful function call\n"""'
},
{
    "instruction": "def parse_query_string(url):\n    from urllib.parse import urlparse, parse_qs\n    parsed = urlparse(url)\n    return parse_qs(parsed.query)",
    "response": '"""Extract query parameters from a URL.\n\nArgs:\n    url: URL string with query parameters\n\nReturns:\n    Dictionary of query parameters\n"""'
},
{
    "instruction": "class Logger:\n    def __init__(self, filename):\n        self.filename = filename\n        self.file = open(filename, 'a')",
    "response": '"""Simple file logger.\n\nAttributes:\n    filename: Name of the log file\n    file: Open file handle for writing\n"""'
},
{
    "instruction": "def compress_whitespace(text):\n    import re\n    return re.sub(r'\\s+', ' ', text).strip()",
    "response": '"""Compress multiple whitespace characters into single spaces.\n\nArgs:\n    text: Input string\n\nReturns:\n    String with normalized whitespace\n"""'
},
{
    "instruction": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "response": '"""Calculate the nth Fibonacci number.\n\nArgs:\n    n: Position in the Fibonacci sequence\n\nReturns:\n    Fibonacci number at position n\n"""'
},
{
    "instruction": "def get_line_count(filepath):\n    with open(filepath, 'r') as f:\n        return sum(1 for line in f)",
    "response": '"""Count the number of lines in a file.\n\nArgs:\n    filepath: Path to the file\n\nReturns:\n    Number of lines as integer\n"""'
},
{
    "instruction": "def create_thumbnail(image_path, size=(128, 128)):\n    from PIL import Image\n    img = Image.open(image_path)\n    img.thumbnail(size)\n    return img",
    "response": '"""Create a thumbnail from an image.\n\nArgs:\n    image_path: Path to the source image\n    size: Tuple of (width, height) for thumbnail\n\nReturns:\n    PIL Image object\n"""'
},
{
    "instruction": "def sanitize_filename(filename):\n    import re\n    return re.sub(r'[<>:\"/\\\\|?*]', '_', filename)",
    "response": '"""Remove invalid characters from a filename.\n\nArgs:\n    filename: Original filename\n\nReturns:\n    Sanitized filename safe for filesystem\n"""'
},
{
    "instruction": "def get_memory_usage():\n    import psutil\n    return psutil.virtual_memory().percent",
    "response": '"""Get current system memory usage percentage.\n\nReturns:\n    Memory usage as percentage float\n"""'
},
{
    "instruction": "def split_into_batches(items, batch_size):\n    for i in range(0, len(items), batch_size):\n        yield items[i:i + batch_size]",
    "response": '"""Split a list into batches of a given size.\n\nArgs:\n    items: List to split\n    batch_size: Size of each batch\n\nYields:\n    Batches of items\n"""'
},
{
    "instruction": "def is_url_valid(url):\n    import re\n    pattern = re.compile(r'^https?://(?:[-\\w.])+(?:[:\\d]+)?(?:/(?:[\\w/_.])*(?:\\?(?:[\\w&=%.])*)?(?:#(?:[\\w.])*)?)?$')\n    return pattern.match(url) is not None",
    "response": '"""Validate if a string is a valid URL.\n\nArgs:\n    url: URL string to validate\n\nReturns:\n    True if valid URL, else False\n"""'
},
{
    "instruction": "def format_bytes(size):\n    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:\n        if size < 1024.0:\n            return f\"{size:.1f} {unit}\"\n        size /= 1024.0\n    return f\"{size:.1f} PB\"",
    "response": '"""Format byte size in human-readable format.\n\nArgs:\n    size: Size in bytes\n\nReturns:\n    Formatted string with appropriate unit\n"""'
},
{
    "instruction": "def deep_get(dictionary, keys):\n    for key in keys:\n        dictionary = dictionary.get(key, {})\n    return dictionary",
    "response": '"""Safely access nested dictionary values.\n\nArgs:\n    dictionary: Nested dictionary\n    keys: List of keys to traverse\n\nReturns:\n    Value at the nested path or empty dict\n"""'
},
{
    "instruction": "def remove_accents(text):\n    import unicodedata\n    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')",
    "response": '"""Remove accents and diacritics from text.\n\nArgs:\n    text: Input string with accents\n\nReturns:\n    String with accents removed\n"""'
},
{
    "instruction": "def get_cpu_count():\n    import multiprocessing\n    return multiprocessing.cpu_count()",
    "response": '"""Get the number of CPU cores available.\n\nReturns:\n    Number of CPU cores as integer\n"""'
},
{
    "instruction": "def interpolate_string(template, **kwargs):\n    return template.format(**kwargs)",
    "response": '"""Interpolate variables into a string template.\n\nArgs:\n    template: String template with placeholders\n    **kwargs: Variables to interpolate\n\nReturns:\n    Interpolated string\n"""'
},
{
    "instruction": "def get_git_commit():\n    import subprocess\n    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()",
    "response": '"""Get the current git commit hash.\n\nReturns:\n    Git commit hash as string\n"""'
},
{
    "instruction": "def levenshtein_distance(s1, s2):\n    if len(s1) < len(s2):\n        return levenshtein_distance(s2, s1)\n    if len(s2) == 0:\n        return len(s1)\n    previous_row = range(len(s2) + 1)\n    for i, c1 in enumerate(s1):\n        current_row = [i + 1]\n        for j, c2 in enumerate(s2):\n            insertions = previous_row[j + 1] + 1\n            deletions = current_row[j] + 1\n            substitutions = previous_row[j] + (c1 != c2)\n            current_row.append(min(insertions, deletions, substitutions))\n        previous_row = current_row\n    return previous_row[-1]",
    "response": '"""Calculate edit distance between two strings.\n\nArgs:\n    s1: First string\n    s2: Second string\n\nReturns:\n    Minimum edit distance as integer\n"""'
},
{
    "instruction": "def create_directory(path):\n    import os\n    os.makedirs(path, exist_ok=True)",
    "response": '"""Create a directory and any necessary parent directories.\n\nArgs:\n    path: Directory path to create\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def get_public_ip():\n    import requests\n    return requests.get('https://api.ipify.org').text",
    "response": '"""Get the public IP address of this machine.\n\nReturns:\n    Public IP address as string\n"""'
},
{
    "instruction": "def debounce(func, delay):\n    import threading\n    timer = None\n    def wrapper(*args, **kwargs):\n        nonlocal timer\n        if timer:\n            timer.cancel()\n        timer = threading.Timer(delay, func, args, kwargs)\n        timer.start()\n    return wrapper",
    "response": '"""Create a debounced version of a function.\n\nArgs:\n    func: Function to debounce\n    delay: Delay in seconds\n\nReturns:\n    Debounced function\n"""'
},
{
    "instruction": "def map_range(value, from_min, from_max, to_min, to_max):\n    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min",
    "response": '"""Map a value from one range to another.\n\nArgs:\n    value: Value to map\n    from_min: Minimum of source range\n    from_max: Maximum of source range\n    to_min: Minimum of target range\n    to_max: Maximum of target range\n\nReturns:\n    Mapped value in target range\n"""'
},
{
    "instruction": "def get_disk_usage(path):\n    import shutil\n    total, used, free = shutil.disk_usage(path)\n    return {'total': total, 'used': used, 'free': free}",
    "response": '"""Get disk usage statistics for a path.\n\nArgs:\n    path: Directory path to check\n\nReturns:\n    Dictionary with total, used, and free bytes\n"""'
},
{
    "instruction": "def celsius_to_fahrenheit(celsius):\n    return celsius * 9/5 + 32",
    "response": '"""Convert temperature from Celsius to Fahrenheit.\n\nArgs:\n    celsius: Temperature in Celsius\n\nReturns:\n    Temperature in Fahrenheit\n"""'
},
{
    "instruction": "def get_matching_files(directory, pattern):\n    import glob\n    import os\n    return glob.glob(os.path.join(directory, pattern))",
    "response": '"""Find files matching a pattern in a directory.\n\nArgs:\n    directory: Directory to search\n    pattern: Glob pattern to match\n\nReturns:\n    List of matching file paths\n"""'
},
{
    "instruction": "def create_zip_archive(files, archive_name):\n    import zipfile\n    with zipfile.ZipFile(archive_name, 'w') as zipf:\n        for file in files:\n            zipf.write(file)",
    "response": '"""Create a ZIP archive from a list of files.\n\nArgs:\n    files: List of file paths to include\n    archive_name: Name of the ZIP file to create\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def generate_password(length=12):\n    import string\n    import random\n    chars = string.ascii_letters + string.digits + '!@#$%^&*'\n    return ''.join(random.choice(chars) for _ in range(length))",
    "response": '"""Generate a random password.\n\nArgs:\n    length: Length of the password\n\nReturns:\n    Random password string\n"""'
},
{
    "instruction": "def get_timezone():\n    import time\n    return time.tzname[0]",
    "response": '"""Get the current system timezone.\n\nReturns:\n    Timezone name as string\n"""'
},
{
    "instruction": "def is_even(number):\n    return number % 2 == 0",
    "response": '"""Check if a number is even.\n\nArgs:\n    number: Integer to check\n\nReturns:\n    True if even, else False\n"""'
},
{
    "instruction": "def download_file(url, filename):\n    import urllib.request\n    urllib.request.urlretrieve(url, filename)",
    "response": '"""Download a file from a URL.\n\nArgs:\n    url: URL of the file to download\n    filename: Local filename to save as\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def validate_json(json_string):\n    import json\n    try:\n        json.loads(json_string)\n        return True\n    except ValueError:\n        return False",
    "response": '"""Validate if a string is valid JSON.\n\nArgs:\n    json_string: String to validate\n\nReturns:\n    True if valid JSON, else False\n"""'
},
{
    "instruction": "def get_process_count():\n    import psutil\n    return len(psutil.pids())",
    "response": '"""Get the number of running processes.\n\nReturns:\n    Number of processes as integer\n"""'
},
{
    "instruction": "def benchmark_function(func, *args, **kwargs):\n    import time\n    start = time.time()\n    result = func(*args, **kwargs)\n    end = time.time()\n    return result, end - start",
    "response": '"""Benchmark the execution time of a function.\n\nArgs:\n    func: Function to benchmark\n    *args: Arguments for the function\n    **kwargs: Keyword arguments for the function\n\nReturns:\n    Tuple of (result, execution_time)\n"""'
},
{
    "instruction": "def find_common_elements(list1, list2):\n    return list(set(list1) & set(list2))",
    "response": '"""Find common elements between two lists.\n\nArgs:\n    list1: First list\n    list2: Second list\n\nReturns:\n    List of common elements\n"""'
},
{
    "instruction": "def get_current_timestamp():\n    import time\n    return int(time.time())",
    "response": '"""Get current Unix timestamp.\n\nReturns:\n    Current timestamp as integer\n"""'
},
{
    "instruction": "def word_wrap(text, width):\n    import textwrap\n    return textwrap.fill(text, width)",
    "response": '"""Wrap text to a specified width.\n\nArgs:\n    text: Text to wrap\n    width: Maximum line width\n\nReturns:\n    Wrapped text string\n"""'
},
{
    "instruction": "class Cache:\n    def __init__(self, max_size=100):\n        self.max_size = max_size\n        self.data = {}",
    "response": '"""Simple in-memory cache with size limit.\n\nAttributes:\n    max_size: Maximum number of items to cache\n    data: Dictionary storing cached items\n"""'
},
{
"instruction": "def normalize_phone_number(phone):\n    import re\n    return re.sub(r'[^0-9]', '', phone)",
"response": '"""Remove all non-digit characters from a phone number.\n\nArgs:\n    phone: Phone number string\n\nReturns:\n    Normalized phone number with digits only\n"""'
},
{
    "instruction": "def get_random_item(items):\n    import random\n    return random.choice(items)",
    "response": '"""Select a random item from a list.\n\nArgs:\n    items: List of items to choose from\n\nReturns:\n    Randomly selected item\n"""'
},
{
    "instruction": "def convert_seconds_to_time(seconds):\n    hours = seconds // 3600\n    minutes = (seconds % 3600) // 60\n    secs = seconds % 60\n    return f\"{hours:02d}:{minutes:02d}:{secs:02d}\"",
    "response": '"""Convert seconds to HH:MM:SS format.\n\nArgs:\n    seconds: Number of seconds\n\nReturns:\n    Time string in HH:MM:SS format\n"""'
},
{
    "instruction": "def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    return n * calculate_factorial(n - 1)",
    "response": '"""Calculate the factorial of a number.\n\nArgs:\n    n: Non-negative integer\n\nReturns:\n    Factorial of n\n"""'
},
{
    "instruction": "def get_file_creation_time(filepath):\n    import os\n    return os.path.getctime(filepath)",
    "response": '"""Get the creation time of a file.\n\nArgs:\n    filepath: Path to the file\n\nReturns:\n    Creation timestamp as float\n"""'
},
{
    "instruction": "def shuffle_list(items):\n    import random\n    shuffled = items.copy()\n    random.shuffle(shuffled)\n    return shuffled",
    "response": '"""Create a shuffled copy of a list.\n\nArgs:\n    items: List to shuffle\n\nReturns:\n    New list with items in random order\n"""'
},
{
    "instruction": "def mask_credit_card(card_number):\n    return '*' * (len(card_number) - 4) + card_number[-4:]",
    "response": '"""Mask a credit card number showing only last 4 digits.\n\nArgs:\n    card_number: Credit card number string\n\nReturns:\n    Masked card number\n"""'
},
{
    "instruction": "def create_slug_from_title(title):\n    import re\n    slug = title.lower()\n    slug = re.sub(r'[^a-z0-9\\s-]', '', slug)\n    slug = re.sub(r'[\\s-]+', '-', slug)\n    return slug.strip('-')",
    "response": '"""Convert a title to a URL-friendly slug.\n\nArgs:\n    title: Title string to convert\n\nReturns:\n    URL-safe slug string\n"""'
},
{
    "instruction": "def count_lines_of_code(filepath):\n    with open(filepath, 'r') as f:\n        return len([line for line in f if line.strip() and not line.strip().startswith('#')])",
    "response": '"""Count non-empty, non-comment lines in a file.\n\nArgs:\n    filepath: Path to the source file\n\nReturns:\n    Number of code lines\n"""'
},
{
    "instruction": "def get_directory_size(directory):\n    import os\n    total_size = 0\n    for dirpath, dirnames, filenames in os.walk(directory):\n        for filename in filenames:\n            filepath = os.path.join(dirpath, filename)\n            total_size += os.path.getsize(filepath)\n    return total_size",
    "response": '"""Calculate the total size of a directory.\n\nArgs:\n    directory: Path to the directory\n\nReturns:\n    Total size in bytes\n"""'
},
{
    "instruction": "def validate_age(age):\n    return isinstance(age, int) and 0 <= age <= 150",
    "response": '"""Validate if an age value is reasonable.\n\nArgs:\n    age: Age value to validate\n\nReturns:\n    True if age is valid, else False\n"""'
},
{
    "instruction": "def extract_numbers_from_text(text):\n    import re\n    return [float(match) for match in re.findall(r'-?\\d+\\.?\\d*', text)]",
    "response": '"""Extract all numbers from a text string.\n\nArgs:\n    text: Input text\n\nReturns:\n    List of numbers found in the text\n"""'
},
{
    "instruction": "def get_most_frequent(items):\n    from collections import Counter\n    counter = Counter(items)\n    return counter.most_common(1)[0][0]",
    "response": '"""Find the most frequently occurring item in a list.\n\nArgs:\n    items: List of items\n\nReturns:\n    Most frequent item\n"""'
},
{
    "instruction": "def create_progress_bar(current, total, width=50):\n    progress = current / total\n    filled = int(width * progress)\n    bar = '█' * filled + '░' * (width - filled)\n    return f\"[{bar}] {progress:.1%}\"",
    "response": '"""Create a text-based progress bar.\n\nArgs:\n    current: Current progress value\n    total: Total value\n    width: Width of the progress bar\n\nReturns:\n    Progress bar string\n"""'
},
{
    "instruction": "def get_user_agent():\n    import platform\n    return f\"MyApp/1.0 ({platform.system()} {platform.release()})\"",
    "response": '"""Generate a user agent string for HTTP requests.\n\nReturns:\n    User agent string\n"""'
},
{
    "instruction": "def calculate_compound_interest(principal, rate, time, compounds_per_year=1):\n    return principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)",
    "response": '"""Calculate compound interest.\n\nArgs:\n    principal: Initial amount\n    rate: Annual interest rate (as decimal)\n    time: Time period in years\n    compounds_per_year: Number of times interest compounds per year\n\nReturns:\n    Final amount after compound interest\n"""'
},
{
    "instruction": "def safe_cast_to_int(value, default=0):\n    try:\n        return int(value)\n    except (ValueError, TypeError):\n        return default",
    "response": '"""Safely convert a value to integer with fallback.\n\nArgs:\n    value: Value to convert\n    default: Default value if conversion fails\n\nReturns:\n    Integer value or default\n"""'
},
{
    "instruction": "def get_similarity_ratio(str1, str2):\n    from difflib import SequenceMatcher\n    return SequenceMatcher(None, str1, str2).ratio()",
    "response": '"""Calculate similarity ratio between two strings.\n\nArgs:\n    str1: First string\n    str2: Second string\n\nReturns:\n    Similarity ratio between 0 and 1\n"""'
},
{
    "instruction": "def create_temp_file(content, suffix='.txt'):\n    import tempfile\n    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:\n        f.write(content)\n        return f.name",
    "response": '"""Create a temporary file with content.\n\nArgs:\n    content: Content to write to the file\n    suffix: File extension\n\nReturns:\n    Path to the temporary file\n"""'
},
{
    "instruction": "def get_screen_resolution():\n    import tkinter as tk\n    root = tk.Tk()\n    width = root.winfo_screenwidth()\n    height = root.winfo_screenheight()\n    root.destroy()\n    return (width, height)",
    "response": '"""Get the screen resolution.\n\nReturns:\n    Tuple of (width, height) in pixels\n"""'
},
{
    "instruction": "def convert_csv_to_json(csv_file, json_file):\n    import csv\n    import json\n    with open(csv_file, 'r') as f:\n        reader = csv.DictReader(f)\n        data = list(reader)\n    with open(json_file, 'w') as f:\n        json.dump(data, f, indent=2)",
    "response": '"""Convert a CSV file to JSON format.\n\nArgs:\n    csv_file: Path to input CSV file\n    json_file: Path to output JSON file\n\nReturns:\n    None\n"""'
},
{
    "instruction": "def get_color_contrast_ratio(color1, color2):\n    def luminance(color):\n        return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]\n    l1 = luminance(color1)\n    l2 = luminance(color2)\n    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)",
    "response": '"""Calculate color contrast ratio between two RGB colors.\n\nArgs:\n    color1: First color as RGB tuple\n    color2: Second color as RGB tuple\n\nReturns:\n    Contrast ratio as float\n"""'
},
{
    "instruction": "def monitor_directory_changes(directory):\n    import os\n    import time\n    last_modified = os.path.getmtime(directory)\n    while True:\n        current_modified = os.path.getmtime(directory)\n        if current_modified != last_modified:\n            yield directory\n            last_modified = current_modified\n        time.sleep(1)",
    "response": '"""Monitor a directory for changes.\n\nArgs:\n    directory: Directory path to monitor\n\nYields:\n    Directory path when changes are detected\n"""'
},
{
    "instruction": "def get_dominant_color(image_path):\n    from PIL import Image\n    from collections import Counter\n    img = Image.open(image_path)\n    img = img.resize((100, 100))\n    colors = img.getdata()\n    most_common = Counter(colors).most_common(1)\n    return most_common[0][0]",
    "response": '"""Get the dominant color from an image.\n\nArgs:\n    image_path: Path to the image file\n\nReturns:\n    RGB tuple of the dominant color\n"""'
},
{
    "instruction": "def calculate_reading_time(text, wpm=200):\n    word_count = len(text.split())\n    minutes = word_count / wpm\n    return max(1, round(minutes))",
    "response": '"""Estimate reading time for a text.\n\nArgs:\n    text: Text to analyze\n    wpm: Words per minute reading speed\n\nReturns:\n    Estimated reading time in minutes\n"""'
},
{
    "instruction": "def get_season(date):\n    month = date.month\n    day = date.day\n    if (month == 12 and day >= 21) or month in [1, 2] or (month == 3 and day < 20):\n        return 'Winter'\n    elif (month == 3 and day >= 20) or month in [4, 5] or (month == 6 and day < 21):\n        return 'Spring'\n    elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day < 23):\n        return 'Summer'\n    else:\n        return 'Fall'",
    "response": '"""Determine the season for a given date.\n\nArgs:\n    date: datetime object\n\nReturns:\n    Season name as string\n"""'
},
{
    "instruction": "def compress_string(text):\n    import zlib\n    import base64\n    compressed = zlib.compress(text.encode())\n    return base64.b64encode(compressed).decode()",
    "response": '"""Compress a string using zlib and base64 encode it.\n\nArgs:\n    text: String to compress\n\nReturns:\n    Compressed and encoded string\n"""'
},
{
    "instruction": "def get_network_interfaces():\n    import psutil\n    return list(psutil.net_if_addrs().keys())",
    "response": '"""Get a list of network interface names.\n\nReturns:\n    List of network interface names\n"""'
},
{
    "instruction": "class EventEmitter:\n    def __init__(self):\n        self.listeners = {}\n    \n    def on(self, event, callback):\n        if event not in self.listeners:\n            self.listeners[event] = []\n        self.listeners[event].append(callback)",
    "response": '"""Simple event emitter for handling callbacks.\n\nAttributes:\n    listeners: Dictionary mapping events to callback lists\n"""'
},
{
    "instruction": "def hash_password(password, salt):\n    import hashlib\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)",
    "response": '"""Hash a password using PBKDF2 with SHA-256.\n\nArgs:\n    password: Plain text password\n    salt: Random salt bytes\n\nReturns:\n    Hashed password as bytes\n"""'
},
{
    "instruction": "def get_file_modified_time(filepath):\n    import os\n    import datetime\n    timestamp = os.path.getmtime(filepath)\n    return datetime.datetime.fromtimestamp(timestamp)",
    "response": '"""Get the last modified time of a file.\n\nArgs:\n    filepath: Path to the file\n\nReturns:\n    datetime object of last modification\n"""'
},
{
    "instruction": "def reverse_string(s):\n    return s[::-1]",
    "response": '"""Reverse a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    Reversed string\n"""'
},
{
    "instruction": "def calculate_percentage(part, total):\n    return (part / total) * 100 if total != 0 else 0",
    "response": '"""Calculate percentage of part relative to total.\n\nArgs:\n    part: Part value\n    total: Total value\n\nReturns:\n    Percentage as float\n"""'
},
{
    "instruction": "def extract_email_domain(email):\n    return email.split('@')[-1]",
    "response": '"""Extract domain from an email address.\n\nArgs:\n    email: Email address string\n\nReturns:\n    Domain part of the email\n"""'
},
{
    "instruction": "def create_date_range(start_date, end_date):\n    import datetime\n    delta = datetime.timedelta(days=1)\n    current = start_date\n    dates = []\n    while current <= end_date:\n        dates.append(current)\n        current += delta\n    return dates",
    "response": '"""Generate a list of dates between two dates.\n\nArgs:\n    start_date: Starting date\n    end_date: Ending date\n\nReturns:\n    List of datetime objects for each day\n"""'
},
{
    "instruction": "def strip_html_comments(html):\n    import re\n    return re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)",
    "response": '"""Remove HTML comments from a string.\n\nArgs:\n    html: HTML string\n\nReturns:\n    HTML with comments removed\n"""'
},
{
    "instruction": "def get_list_intersection(list1, list2):\n    return list(set(list1).intersection(set(list2)))",
    "response": '"""Find intersection of two lists.\n\nArgs:\n    list1: First list\n    list2: Second list\n\nReturns:\n    List of common elements\n"""'
},
{
    "instruction": "def format_phone_number(phone):\n    digits = ''.join(filter(str.isdigit, phone))\n    if len(digits) == 10:\n        return f\"({digits[:3]}) {digits[3:6]}-{digits[6:]}\"\n    return phone",
    "response": '"""Format a phone number to (XXX) XXX-XXXX format.\n\nArgs:\n    phone: Phone number string\n\nReturns:\n    Formatted phone number\n"""'
},
{
    "instruction": "def get_workdays_count(start_date, end_date):\n    import datetime\n    count = 0\n    current = start_date\n    while current <= end_date:\n        if current.weekday() < 5:  # Monday=0, Sunday=6\n            count += 1\n        current += datetime.timedelta(days=1)\n    return count",
    "response": '"""Count weekdays between two dates.\n\nArgs:\n    start_date: Starting date\n    end_date: Ending date\n\nReturns:\n    Number of weekdays as integer\n"""'
},
{
    "instruction": "def truncate_string_words(text, max_words):\n    words = text.split()\n    if len(words) <= max_words:\n        return text\n    return ' '.join(words[:max_words]) + '...'",
    "response": '"""Truncate text to a maximum number of words.\n\nArgs:\n    text: Input text\n    max_words: Maximum number of words\n\nReturns:\n    Truncated text string\n"""'
},
{
    "instruction": "def convert_bytes_to_mb(bytes_value):\n    return bytes_value / (1024 * 1024)",
    "response": '"""Convert bytes to megabytes.\n\nArgs:\n    bytes_value: Size in bytes\n\nReturns:\n    Size in megabytes as float\n"""'
},
{
    "instruction": "def validate_credit_card_luhn(card_number):\n    digits = [int(d) for d in str(card_number) if d.isdigit()]\n    for i in range(len(digits) - 2, -1, -2):\n        digits[i] *= 2\n        if digits[i] > 9:\n            digits[i] -= 9\n    return sum(digits) % 10 == 0",
    "response": '"""Validate credit card number using Luhn algorithm.\n\nArgs:\n    card_number: Credit card number\n\nReturns:\n    True if valid, else False\n"""'
},
{
    "instruction": "def get_dict_keys_by_value(dictionary, target_value):\n    return [key for key, value in dictionary.items() if value == target_value]",
    "response": '"""Find dictionary keys that have a specific value.\n\nArgs:\n    dictionary: Dictionary to search\n    target_value: Value to find\n\nReturns:\n    List of keys with the target value\n"""'
},
{
    "instruction": "def calculate_distance_2d(point1, point2):\n    import math\n    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)",
    "response": '"""Calculate Euclidean distance between two 2D points.\n\nArgs:\n    point1: First point as (x, y) tuple\n    point2: Second point as (x, y) tuple\n\nReturns:\n    Distance as float\n"""'
},
{
    "instruction": "def create_nested_dict(keys, value):\n    result = {}\n    current = result\n    for key in keys[:-1]:\n        current[key] = {}\n        current = current[key]\n    current[keys[-1]] = value\n    return result",
    "response": '"""Create a nested dictionary from a list of keys.\n\nArgs:\n    keys: List of keys for nesting\n    value: Value to assign to the final key\n\nReturns:\n    Nested dictionary\n"""'
},
{
    "instruction": "def get_string_checksum(text):\n    import hashlib\n    return hashlib.md5(text.encode()).hexdigest()[:8]",
    "response": '"""Generate a short checksum for a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    8-character checksum\n"""'
},
{
    "instruction": "def find_longest_word(text):\n    words = text.split()\n    return max(words, key=len) if words else ''",
    "response": '"""Find the longest word in a text.\n\nArgs:\n    text: Input text\n\nReturns:\n    Longest word as string\n"""'
},
{
    "instruction": "def convert_dict_to_query_string(params):\n    from urllib.parse import urlencode\n    return urlencode(params)",
    "response": '"""Convert dictionary to URL query string.\n\nArgs:\n    params: Dictionary of parameters\n\nReturns:\n    URL-encoded query string\n"""'
},
{
    "instruction": "def get_fibonacci_sequence(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    sequence = [0, 1]\n    for i in range(2, n):\n        sequence.append(sequence[i-1] + sequence[i-2])\n    return sequence",
    "response": '"""Generate Fibonacci sequence up to n terms.\n\nArgs:\n    n: Number of terms to generate\n\nReturns:\n    List of Fibonacci numbers\n"""'
},
{
    "instruction": "def strip_non_alphanumeric(text):\n    import re\n    return re.sub(r'[^a-zA-Z0-9]', '', text)",
    "response": '"""Remove all non-alphanumeric characters from text.\n\nArgs:\n    text: Input string\n\nReturns:\n    String with only letters and numbers\n"""'
},
{
    "instruction": "def calculate_circle_area(radius):\n    import math\n    return math.pi * radius ** 2",
    "response": '"""Calculate the area of a circle.\n\nArgs:\n    radius: Radius of the circle\n\nReturns:\n    Area of the circle\n"""'
},
{
    "instruction": "def find_files_by_extension(directory, extension):\n    import os\n    matches = []\n    for root, dirs, files in os.walk(directory):\n        for file in files:\n            if file.endswith(extension):\n                matches.append(os.path.join(root, file))\n    return matches",
    "response": '"""Find all files with a specific extension in a directory tree.\n\nArgs:\n    directory: Root directory to search\n    extension: File extension to match\n\nReturns:\n    List of file paths\n"""'
},
{
    "instruction": "def get_average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0",
    "response": '"""Calculate the average of a list of numbers.\n\nArgs:\n    numbers: List of numeric values\n\nReturns:\n    Average as float\n"""'
},
{
    "instruction": "def encode_url(url):\n    from urllib.parse import quote\n    return quote(url, safe=':/?#[]@!$&\\'()*+,;=')",
    "response": '"""URL encode a string while preserving URL structure.\n\nArgs:\n    url: URL string to encode\n\nReturns:\n    URL-encoded string\n"""'
},
{
    "instruction": "def get_list_mode(items):\n    from collections import Counter\n    if not items:\n        return None\n    counts = Counter(items)\n    max_count = max(counts.values())\n    modes = [item for item, count in counts.items() if count == max_count]\n    return modes[0] if len(modes) == 1 else modes",
    "response": '"""Find the most frequent item(s) in a list.\n\nArgs:\n    items: List of items\n\nReturns:\n    Most frequent item or list of items if tied\n"""'
},
{
    "instruction": "def create_backup_filename(original_path):\n    import os\n    import datetime\n    base, ext = os.path.splitext(original_path)\n    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')\n    return f\"{base}_{timestamp}{ext}\"",
    "response": '"""Generate a timestamped backup filename.\n\nArgs:\n    original_path: Original file path\n\nReturns:\n    Backup filename with timestamp\n"""'
},
{
    "instruction": "def split_name(full_name):\n    parts = full_name.strip().split()\n    if len(parts) == 1:\n        return parts[0], ''\n    elif len(parts) == 2:\n        return parts[0], parts[1]\n    else:\n        return parts[0], ' '.join(parts[1:])",
    "response": '"""Split a full name into first and last name.\n\nArgs:\n    full_name: Full name string\n\nReturns:\n    Tuple of (first_name, last_name)\n"""'
},
{
    "instruction": "def get_unique_characters(text):\n    return ''.join(set(text))",
    "response": '"""Get unique characters from a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    String containing unique characters\n"""'
},
{
    "instruction": "def calculate_tax(amount, tax_rate):\n    return amount * (tax_rate / 100)",
    "response": '"""Calculate tax amount.\n\nArgs:\n    amount: Base amount\n    tax_rate: Tax rate as percentage\n\nReturns:\n    Tax amount as float\n"""'
},
{
    "instruction": "def find_substring_positions(text, substring):\n    positions = []\n    start = 0\n    while True:\n        pos = text.find(substring, start)\n        if pos == -1:\n            break\n        positions.append(pos)\n        start = pos + 1\n    return positions",
    "response": '"""Find all positions of a substring in text.\n\nArgs:\n    text: Text to search in\n    substring: Substring to find\n\nReturns:\n    List of starting positions\n"""'
},
{
    "instruction": "def convert_temperature_scale(temp, from_scale, to_scale):\n    if from_scale == 'C' and to_scale == 'F':\n        return (temp * 9/5) + 32\n    elif from_scale == 'F' and to_scale == 'C':\n        return (temp - 32) * 5/9\n    elif from_scale == 'C' and to_scale == 'K':\n        return temp + 273.15\n    elif from_scale == 'K' and to_scale == 'C':\n        return temp - 273.15\n    else:\n        return temp",
    "response": '"""Convert temperature between different scales.\n\nArgs:\n    temp: Temperature value\n    from_scale: Source scale (C, F, K)\n    to_scale: Target scale (C, F, K)\n\nReturns:\n    Converted temperature\n"""'
},
{
    "instruction": "def get_html_text(html):\n    import re\n    text = re.sub(r'<[^>]+>', '', html)\n    return text.strip()",
    "response": '"""Extract text content from HTML.\n\nArgs:\n    html: HTML string\n\nReturns:\n    Plain text content\n"""'
},
{
    "instruction": "def calculate_rectangle_area(width, height):\n    return width * height",
    "response": '"""Calculate the area of a rectangle.\n\nArgs:\n    width: Width of the rectangle\n    height: Height of the rectangle\n\nReturns:\n    Area of the rectangle\n"""'
},
{
    "instruction": "def merge_sorted_lists(list1, list2):\n    merged = []\n    i = j = 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            merged.append(list1[i])\n            i += 1\n        else:\n            merged.append(list2[j])\n            j += 1\n    merged.extend(list1[i:])\n    merged.extend(list2[j:])\n    return merged",
    "response": '"""Merge two sorted lists into one sorted list.\n\nArgs:\n    list1: First sorted list\n    list2: Second sorted list\n\nReturns:\n    Merged sorted list\n"""'
},
{
    "instruction": "def get_file_basename(filepath):\n    import os\n    return os.path.basename(filepath)",
    "response": '"""Get the filename from a file path.\n\nArgs:\n    filepath: Full file path\n\nReturns:\n    Filename without directory path\n"""'
},
{
    "instruction": "def count_character_frequency(text):\n    from collections import Counter\n    return dict(Counter(text.lower()))",
    "response": '"""Count frequency of each character in text.\n\nArgs:\n    text: Input text\n\nReturns:\n    Dictionary mapping characters to frequencies\n"""'
},
{
    "instruction": "def create_multiplier_function(factor):\n    def multiplier(x):\n        return x * factor\n    return multiplier",
    "response": '"""Create a function that multiplies by a given factor.\n\nArgs:\n    factor: Multiplication factor\n\nReturns:\n    Function that multiplies input by factor\n"""'
},
{
    "instruction": "def get_median(numbers):\n    sorted_nums = sorted(numbers)\n    n = len(sorted_nums)\n    if n % 2 == 0:\n        return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2\n    else:\n        return sorted_nums[n//2]",
    "response": '"""Calculate the median of a list of numbers.\n\nArgs:\n    numbers: List of numeric values\n\nReturns:\n    Median value as float\n"""'
},
{
    "instruction": "def validate_ip_address(ip):\n    import re\n    pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'\n    return re.match(pattern, ip) is not None",
    "response": '"""Validate IPv4 address format.\n\nArgs:\n    ip: IP address string\n\nReturns:\n    True if valid IPv4, else False\n"""'
},
{
    "instruction": "def get_timestamp_difference(timestamp1, timestamp2):\n    return abs(timestamp1 - timestamp2)",
    "response": '"""Calculate absolute difference between two timestamps.\n\nArgs:\n    timestamp1: First timestamp\n    timestamp2: Second timestamp\n\nReturns:\n    Absolute difference in seconds\n"""'
},
{
    "instruction": "def wrap_text_at_column(text, column=80):\n    import textwrap\n    return '\\n'.join(textwrap.wrap(text, width=column))",
    "response": '"""Wrap text at specified column width.\n\nArgs:\n    text: Text to wrap\n    column: Column width for wrapping\n\nReturns:\n    Text wrapped at specified width\n"""'
},
{
    "instruction": "def get_list_standard_deviation(numbers):\n    import math\n    if len(numbers) < 2:\n        return 0\n    mean = sum(numbers) / len(numbers)\n    variance = sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)\n    return math.sqrt(variance)",
    "response": '"""Calculate standard deviation of a list of numbers.\n\nArgs:\n    numbers: List of numeric values\n\nReturns:\n    Standard deviation as float\n"""'
},
{
    "instruction": "def remove_empty_strings(string_list):\n    return [s for s in string_list if s.strip()]",
    "response": '"""Remove empty or whitespace-only strings from a list.\n\nArgs:\n    string_list: List of strings\n\nReturns:\n    List with empty strings removed\n"""'
},
{
    "instruction": "def get_bytes_from_file(filepath):\n    with open(filepath, 'rb') as f:\n        return f.read()",
    "response": '"""Read file content as bytes.\n\nArgs:\n    filepath: Path to the file\n\nReturns:\n    File content as bytes\n"""'
},
{
    "instruction": "def calculate_hypotenuse(a, b):\n    import math\n    return math.sqrt(a**2 + b**2)",
    "response": '"""Calculate hypotenuse of a right triangle.\n\nArgs:\n    a: Length of first side\n    b: Length of second side\n\nReturns:\n    Length of hypotenuse\n"""'
},
{
    "instruction": "def get_string_similarity(str1, str2):\n    from difflib import SequenceMatcher\n    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()",
    "response": '"""Calculate similarity between two strings (case-insensitive).\n\nArgs:\n    str1: First string\n    str2: Second string\n\nReturns:\n    Similarity ratio between 0 and 1\n"""'
},
{
    "instruction": "def create_counter_function():\n    count = 0\n    def counter():\n        nonlocal count\n        count += 1\n        return count\n    return counter",
    "response": '"""Create a counter function that increments on each call.\n\nReturns:\n    Function that returns incremented count\n"""'
},
{
    "instruction": "def parse_csv_line(line, delimiter=','):\n    import csv\n    return list(csv.reader([line], delimiter=delimiter))[0]",
    "response": '"""Parse a single CSV line into fields.\n\nArgs:\n    line: CSV line string\n    delimiter: Field delimiter\n\nReturns:\n    List of parsed fields\n"""'
},
{
    "instruction": "class SimpleQueue:\n    def __init__(self):\n        self.items = []\n    \n    def enqueue(self, item):\n        self.items.append(item)\n    \n    def dequeue(self):\n        return self.items.pop(0) if self.items else None",
    "response": '"""Simple FIFO queue implementation.\n\nAttributes:\n    items: List storing queue items\n"""'
},
{
    "instruction": "def invert_boolean(flag):\n    return not flag",
    "response": '"""Invert a boolean value.\n\nArgs:\n    flag: Boolean value\n\nReturns:\n    Opposite of the input boolean\n"""'
},
{
    "instruction": "def is_multiple(x, base):\n    return x % base == 0",
    "response": '"""Check if a number is a multiple of another.\n\nArgs:\n    x: Number to check\n    base: Base number\n\nReturns:\n    True if x is a multiple of base, else False\n"""'
},
{
    "instruction": "def get_second_item(items):\n    return items[1] if len(items) > 1 else None",
    "response": '"""Return the second item in a list if available.\n\nArgs:\n    items: List of elements\n\nReturns:\n    Second element or None\n"""'
},
{
    "instruction": "def starts_with_vowel(word):\n    return word[0].lower() in 'aeiou'",
    "response": '"""Check if a word starts with a vowel.\n\nArgs:\n    word: Input word\n\nReturns:\n    True if starts with a vowel, else False\n"""'
},
{
    "instruction": "def int_to_bin(n):\n    return bin(n)[2:]",
    "response": '"""Convert an integer to a binary string.\n\nArgs:\n    n: Integer to convert\n\nReturns:\n    Binary string without prefix\n"""'
},
{
    "instruction": "def repeat_list(values, times):\n    return values * times",
    "response": '"""Repeat a list multiple times.\n\nArgs:\n    values: List to repeat\n    times: Number of repetitions\n\nReturns:\n    Repeated list\n"""'
},
{
    "instruction": "def is_uppercase(s):\n    return s.isupper()",
    "response": '"""Check if a string is fully uppercase.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if all characters are uppercase\n"""'
},
{
    "instruction": "def remove_trailing_zeros(number_str):\n    return number_str.rstrip('0').rstrip('.') if '.' in number_str else number_str",
    "response": '"""Remove trailing zeros from a decimal string.\n\nArgs:\n    number_str: String representing a decimal number\n\nReturns:\n    Cleaned number string\n"""'
},
{
    "instruction": "def sum_of_digits(n):\n    return sum(int(d) for d in str(abs(n)))",
    "response": '"""Calculate the sum of digits in an integer.\n\nArgs:\n    n: Input integer\n\nReturns:\n    Sum of digits as integer\n"""'
},
{
    "instruction": "def strip_quotes(text):\n    return text.strip('\"\\'')",
    "response": '"""Remove surrounding quotes from a string.\n\nArgs:\n    text: String with potential quotes\n\nReturns:\n    Unquoted string\n"""'
},
{
    "instruction": "def contains_substring(text, query):\n    return query in text",
    "response": '"""Check if a substring exists in a string.\n\nArgs:\n    text: Source string\n    query: Substring to look for\n\nReturns:\n    True if query is in text\n"""'
},
{
    "instruction": "def pluralize(word, count):\n    return word if count == 1 else word + 's'",
    "response": '"""Return word in singular or plural based on count.\n\nArgs:\n    word: Word to pluralize\n    count: Associated quantity\n\nReturns:\n    Singular or plural form of the word\n"""'
},
{
    "instruction": "def char_frequency(text):\n    return {c: text.count(c) for c in set(text)}",
    "response": '"""Calculate frequency of each character in text.\n\nArgs:\n    text: Input string\n\nReturns:\n    Dictionary of character frequencies\n"""'
},
{
    "instruction": "def list_head_tail(lst):\n    return (lst[0], lst[-1]) if lst else (None, None)",
    "response": '"""Get the first and last items in a list.\n\nArgs:\n    lst: Input list\n\nReturns:\n    Tuple of (first, last) or (None, None) if empty\n"""'
},
{
    "instruction": "def escape_quotes(text):\n    return text.replace('\"', '\\\\\"').replace(\"'\", \"\\\\'\")",
    "response": '"""Escape quotes in a string.\n\nArgs:\n    text: Input text\n\nReturns:\n    Escaped string\n"""'
},
{
    "instruction": "def clamp_list(values, low, high):\n    return [max(low, min(v, high)) for v in values]",
    "response": '"""Clamp all values in a list within a range.\n\nArgs:\n    values: List of numeric values\n    low: Minimum allowed value\n    high: Maximum allowed value\n\nReturns:\n    List of clamped values\n"""'
},
{
    "instruction": "def format_name(first, last):\n    return f\"{first.capitalize()} {last.capitalize()}\"",
    "response": '"""Format first and last name with capitalization.\n\nArgs:\n    first: First name\n    last: Last name\n\nReturns:\n    Formatted full name\n"""'
},
{
    "instruction": "def swap_case(text):\n    return text.swapcase()",
    "response": '"""Invert the case of each character in a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    String with case of each character swapped\n"""'
},
{
    "instruction": "def is_sorted_asc(lst):\n    return all(x <= y for x, y in zip(lst, lst[1:]))",
    "response": '"""Check if a list is sorted in ascending order.\n\nArgs:\n    lst: List of comparable elements\n\nReturns:\n    True if sorted, else False\n"""'
},
{
    "instruction": "def repeat_char(char, n):\n    return char * n",
    "response": '"""Repeat a character n times.\n\nArgs:\n    char: Character to repeat\n    n: Number of repetitions\n\nReturns:\n    Repeated string\n"""'
},
{
    "instruction": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
    "response": '"""Check if a number is a prime.\n\nArgs:\n    n: Integer to check\n\nReturns:\n    True if prime, else False\n"""'
},
{
    "instruction": "def safe_get(d, key, default=None):\n    return d[key] if key in d else default",
    "response": '"""Safely get a value from a dictionary.\n\nArgs:\n    d: Dictionary\n    key: Key to look up\n    default: Value to return if key is missing\n\nReturns:\n    Value from dict or default\n"""'
},
{
    "instruction": "def median(a, b, c):\n    return sorted([a, b, c])[1]",
    "response": '"""Return the median of three numbers.\n\nArgs:\n    a: First number\n    b: Second number\n    c: Third number\n\nReturns:\n    Median value\n"""'
},
{
    "instruction": "def split_path(path):\n    return path.rsplit('/', 1)",
    "response": '"""Split a file path into directory and filename.\n\nArgs:\n    path: File path string\n\nReturns:\n    List of [directory, filename]\n"""'
},
{
    "instruction": "def ends_with(text, suffix):\n    return text.endswith(suffix)",
    "response": '"""Check if text ends with a given suffix.\n\nArgs:\n    text: Input string\n    suffix: Ending string to check\n\nReturns:\n    True if text ends with suffix\n"""'
},
{
    "instruction": "def round_down(n, decimals=0):\n    multiplier = 10 ** decimals\n    return math.floor(n * multiplier) / multiplier",
    "response": '"""Round a number down to a specified number of decimals.\n\nArgs:\n    n: Number to round\n    decimals: Decimal places\n\nReturns:\n    Rounded down number\n"""'
},
{
    "instruction": "def get_percent_change(old, new):\n    return ((new - old) / old) * 100 if old != 0 else None",
    "response": '"""Calculate percentage change from old to new value.\n\nArgs:\n    old: Original value\n    new: Updated value\n\nReturns:\n    Percentage change, or None if old is zero\n"""'
},
{
    "instruction": "def make_range(start, end):\n    return list(range(start, end + 1))",
    "response": '"""Generate a list of integers from start to end inclusive.\n\nArgs:\n    start: Start of range\n    end: End of range\n\nReturns:\n    List of integers\n"""'
},
{
    "instruction": "def kebab_to_snake(text):\n    return text.replace('-', '_')",
    "response": '"""Convert kebab-case string to snake_case.\n\nArgs:\n    text: Input string in kebab-case\n\nReturns:\n    String in snake_case\n"""'
},
{
    "instruction": "def is_ascii(s):\n    return all(ord(c) < 128 for c in s)",
    "response": '"""Check if all characters are ASCII.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if all characters are ASCII\n"""'
},
{
    "instruction": "def zero_pad(n, width):\n    return str(n).zfill(width)",
    "response": '"""Pad a number with leading zeros.\n\nArgs:\n    n: Number to format\n    width: Total width of resulting string\n\nReturns:\n    Zero-padded string\n"""'
},
{
    "instruction": "def join_words(words, sep=', '):\n    return sep.join(words)",
    "response": '"""Join a list of words with a separator.\n\nArgs:\n    words: List of strings\n    sep: Separator string\n\nReturns:\n    Concatenated string\n"""'
},
{
    "instruction": "def contains_digit(s):\n    return any(c.isdigit() for c in s)",
    "response": '"""Check if a string contains any digits.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if any character is a digit\n"""'
},
{
    "instruction": "def same_type(a, b):\n    return type(a) == type(b)",
    "response": '"""Check whether two objects are of the same type.\n\nArgs:\n    a: First object\n    b: Second object\n\nReturns:\n    True if both are same type\n"""'
},
{
    "instruction": "def truncate_words(text, n):\n    return ' '.join(text.split()[:n])",
    "response": '"""Truncate text to a given number of words.\n\nArgs:\n    text: Input string\n    n: Maximum number of words\n\nReturns:\n    Truncated string\n"""'
},
{
    "instruction": "def absolute_diff(a, b):\n    return abs(a - b)",
    "response": '"""Calculate absolute difference between two numbers.\n\nArgs:\n    a: First number\n    b: Second number\n\nReturns:\n    Absolute difference\n"""'
},
{
    "instruction": "def is_divisible(x, y):\n    return y != 0 and x % y == 0",
    "response": '"""Check if x is divisible by y.\n\nArgs:\n    x: Dividend\n    y: Divisor\n\nReturns:\n    True if divisible, else False\n"""'
},
{
    "instruction": "def square_root(n):\n    return n ** 0.5",
    "response": '"""Compute square root of a number.\n\nArgs:\n    n: Number to compute square root of\n\nReturns:\n    Square root value\n"""'
},
{
    "instruction": "def has_lowercase(text):\n    return any(c.islower() for c in text)",
    "response": '"""Check if a string contains lowercase letters.\n\nArgs:\n    text: Input string\n\nReturns:\n    True if at least one lowercase letter exists\n"""'
},
{
    "instruction": "def index_of_max(values):\n    return max(enumerate(values), key=lambda x: x[1])[0]",
    "response": '"""Find index of the maximum value in a list.\n\nArgs:\n    values: List of numeric values\n\nReturns:\n    Index of the maximum value\n"""'
},
{
    "instruction": "def reverse_words(text):\n    return ' '.join(reversed(text.split()))",
    "response": '"""Reverse the order of words in a string.\n\nArgs:\n    text: Input sentence\n\nReturns:\n    String with words in reverse order\n"""'
},
{
    "instruction": "def pad_left(s, width):\n    return s.rjust(width)",
    "response": '"""Pad a string on the left with spaces.\n\nArgs:\n    s: String to pad\n    width: Total width after padding\n\nReturns:\n    Left-padded string\n"""'
},
{
    "instruction": "def normalize_whitespace(text):\n    return ' '.join(text.split())",
    "response": '"""Remove extra whitespace from a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    String with normalized spacing\n"""'
},
{
    "instruction": "def capitalize_sentences(paragraph):\n    return '. '.join(s.strip().capitalize() for s in paragraph.split('.'))",
    "response": '"""Capitalize the first letter of each sentence.\n\nArgs:\n    paragraph: String containing sentences\n\nReturns:\n    Text with capitalized sentences\n"""'
},
{
    "instruction": "def first_non_empty(*args):\n    for arg in args:\n        if arg:\n            return arg",
    "response": '"""Return the first non-empty argument.\n\nArgs:\n    *args: Variable-length arguments\n\nReturns:\n    First non-empty value or None\n"""'
},
{
    "instruction": "def count_lines(text):\n    return len(text.splitlines())",
    "response": '"""Count the number of lines in a string.\n\nArgs:\n    text: Multiline string\n\nReturns:\n    Number of lines\n"""'
},
{
    "instruction": "def get_even_numbers(numbers):\n    return [n for n in numbers if n % 2 == 0]",
    "response": '"""Filter even numbers from a list.\n\nArgs:\n    numbers: List of integers\n\nReturns:\n    List of even integers\n"""'
},
{
    "instruction": "def remove_none(items):\n    return [item for item in items if item is not None]",
    "response": '"""Remove None values from a list.\n\nArgs:\n    items: List containing elements\n\nReturns:\n    List without None values\n"""'
},
{
    "instruction": "def contains_only_letters(s):\n    return s.isalpha()",
    "response": '"""Check if a string contains only alphabetic characters.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if all characters are letters\n"""'
},
{
    "instruction": "def count_char(text, char):\n    return text.count(char)",
    "response": '"""Count occurrences of a character in a string.\n\nArgs:\n    text: Input string\n    char: Character to count\n\nReturns:\n    Number of occurrences\n"""'
},
{
    "instruction": "def merge_dicts(a, b):\n    return {**a, **b}",
    "response": '"""Merge two dictionaries.\n\nArgs:\n    a: First dictionary\n    b: Second dictionary\n\nReturns:\n    Combined dictionary with b overriding a\n"""'
},
{
    "instruction": "def to_title_case(text):\n    return text.title()",
    "response": '"""Convert string to title case.\n\nArgs:\n    text: Input string\n\nReturns:\n    String with each word capitalized\n"""'
},
{
    "instruction": "def count_uppercase(s):\n    return sum(1 for c in s if c.isupper())",
    "response": '"""Count uppercase letters in a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    Number of uppercase characters\n"""'
},
{
    "instruction": "def is_all_whitespace(s):\n    return s.strip() == ''",
    "response": '"""Check if a string contains only whitespace.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if only whitespace, else False\n"""'
},
{
    "instruction": "def to_ascii_codes(text):\n    return [ord(c) for c in text]",
    "response": '"""Convert a string to list of ASCII codes.\n\nArgs:\n    text: Input string\n\nReturns:\n    List of ASCII integer values\n"""'
},
{
    "instruction": "def xor_booleans(a, b):\n    return bool(a) ^ bool(b)",
    "response": '"""Perform logical XOR on two booleans.\n\nArgs:\n    a: First boolean\n    b: Second boolean\n\nReturns:\n    True if only one is True\n"""'
},
{
    "instruction": "def digits_only(text):\n    return ''.join(filter(str.isdigit, text))",
    "response": '"""Extract only digits from a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    String of digits only\n"""'
},
{
    "instruction": "def square_elements(lst):\n    return [x**2 for x in lst]",
    "response": '"""Square each element in a list.\n\nArgs:\n    lst: List of numeric values\n\nReturns:\n    List of squared values\n"""'
},
{
    "instruction": "def has_uppercase(s):\n    return any(c.isupper() for c in s)",
    "response": '"""Check if a string contains any uppercase letters.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if any character is uppercase\n"""'
},
{
    "instruction": "def get_last_n_items(lst, n):\n    return lst[-n:]",
    "response": '"""Get the last n elements of a list.\n\nArgs:\n    lst: List of items\n    n: Number of items to return from end\n\nReturns:\n    List of last n items\n"""'
},
{
    "instruction": "def foo(x):\n    return x[::-1]",
    "response": '"""Reverse a sequence.\n\nArgs:\n    x: List or string to reverse\n\nReturns:\n    Reversed sequence\n"""'
},
{
    "instruction": "def do_stuff(a, b):\n    return a if a < b else b",
    "response": '"""Return the smaller of two values.\n\nArgs:\n    a: First value\n    b: Second value\n\nReturns:\n    Minimum of the two inputs\n"""'
},
{
    "instruction": "def thing(n):\n    return n * n",
    "response": '"""Square a number.\n\nArgs:\n    n: Number to square\n\nReturns:\n    Square of the input\n"""'
},
{
    "instruction": "def do_the_thing(text):\n    return text.strip().lower()",
    "response": '"""Normalize a string by stripping and lowercasing.\n\nArgs:\n    text: Input string\n\nReturns:\n    Normalized string\n"""'
},
{
    "instruction": "def stuffify(x):\n    return sum(x) / len(x)",
    "response": '"""Calculate the mean of a list.\n\nArgs:\n    x: List of numbers\n\nReturns:\n    Average value\n"""'
},
{
    "instruction": "def process_it(word):\n    return word[0].upper() + word[1:].lower()",
    "response": '"""Capitalize the first letter of a word.\n\nArgs:\n    word: Input string\n\nReturns:\n    Word with capitalized first letter\n"""'
},
{
    "instruction": "def blob(a, b):\n    return abs(a - b)",
    "response": '"""Compute the absolute difference between two values.\n\nArgs:\n    a: First number\n    b: Second number\n\nReturns:\n    Absolute difference\n"""'
},
{
    "instruction": "def what(values):\n    return values[-1]",
    "response": '"""Get the last element from a list.\n\nArgs:\n    values: List of elements\n\nReturns:\n    Last item in the list\n"""'
},
{
    "instruction": "def thinger(text, prefix):\n    return text.startswith(prefix)",
    "response": '"""Check if a string starts with a given prefix.\n\nArgs:\n    text: Input string\n    prefix: Prefix to check\n\nReturns:\n    True if text starts with prefix\n"""'
},
{
    "instruction": "def glorp(numbers):\n    return min(numbers)",
    "response": '"""Return the minimum value in a list.\n\nArgs:\n    numbers: List of numeric values\n\nReturns:\n    Minimum number\n"""'
},
{
    "instruction": "def wobble(x):\n    return x.strip().split()",
    "response": '"""Split a string into words after stripping whitespace.\n\nArgs:\n    x: Input string\n\nReturns:\n    List of words\n"""'
},
{
    "instruction": "def zoomy(lst):\n    return [i for i in lst if i]",
    "response": '"""Remove falsy values from a list.\n\nArgs:\n    lst: List of elements\n\nReturns:\n    Filtered list with only truthy values\n"""'
},
{
    "instruction": "def meh(n):\n    return n % 2 == 0",
    "response": '"""Check if a number is even.\n\nArgs:\n    n: Integer to check\n\nReturns:\n    True if even, else False\n"""'
},
{
    "instruction": "def quack(a):\n    return list(str(a))",
    "response": '"""Convert an input into a list of characters.\n\nArgs:\n    a: Input convertible to string\n\nReturns:\n    List of characters\n"""'
},
{
    "instruction": "def widget(v):\n    return v ** 0.5",
    "response": '"""Calculate the square root of a number.\n\nArgs:\n    v: Input number\n\nReturns:\n    Square root\n"""'
},
{
    "instruction": "def frob(x):\n    return sorted(set(x))",
    "response": '"""Return sorted unique elements from a list.\n\nArgs:\n    x: List of elements\n\nReturns:\n    Sorted list of unique values\n"""'
},
{
    "instruction": "def thingamajig(lst):\n    return len(set(lst)) == len(lst)",
    "response": '"""Check if all elements in a list are unique.\n\nArgs:\n    lst: List of elements\n\nReturns:\n    True if no duplicates\n"""'
},
{
    "instruction": "def whoknows(path):\n    return path.endswith(('.jpg', '.png'))",
    "response": '"""Check if a path has an image file extension.\n\nArgs:\n    path: File path string\n\nReturns:\n    True if path ends in .jpg or .png\n"""'
},
{
    "instruction": "def do_it_all(s):\n    return ''.join(sorted(set(s)))",
    "response": '"""Return a sorted string of unique characters.\n\nArgs:\n    s: Input string\n\nReturns:\n    Sorted unique characters\n"""'
},
{
    "instruction": "def build_full_name(first, middle, last, suffix=None):\n    parts = [first, middle, last]\n    if suffix:\n        parts.append(suffix)\n    return ' '.join(p for p in parts if p)",
    "response": '"""Construct a full name from components.\n\nArgs:\n    first: First name\n    middle: Middle name\n    last: Last name\n    suffix: Optional suffix\n\nReturns:\n    Full name as a string\n"""'
},
{
    "instruction": "def calculate_discount(price, rate, tax, quantity):\n    subtotal = price * quantity\n    taxed = subtotal * (1 + tax)\n    return taxed * (1 - rate)",
    "response": '"""Calculate discounted total price with tax.\n\nArgs:\n    price: Price per item\n    rate: Discount rate (0–1)\n    tax: Tax rate (0–1)\n    quantity: Number of items\n\nReturns:\n    Final price after tax and discount\n"""'
},
{
    "instruction": "def replace_substring(text, old, new, max_replacements=1):\n    return text.replace(old, new, max_replacements)",
    "response": '"""Replace a substring in text with optional max count.\n\nArgs:\n    text: Source string\n    old: Substring to replace\n    new: Replacement string\n    max_replacements: Maximum number of replacements\n\nReturns:\n    Modified string\n"""'
},
{
    "instruction": "def create_email(to, subject, body, cc=None, bcc=None):\n    return {\n        'to': to,\n        'subject': subject,\n        'body': body,\n        'cc': cc or [],\n        'bcc': bcc or []\n    }",
    "response": '"""Create a basic email structure.\n\nArgs:\n    to: List of recipients\n    subject: Email subject\n    body: Email content\n    cc: Optional CC recipients\n    bcc: Optional BCC recipients\n\nReturns:\n    Dictionary representing the email\n"""'
},
{
    "instruction": "def filter_range(values, low, high, inclusive=True, round_result=False):\n    filtered = [v for v in values if (low <= v <= high) if inclusive else (low < v < high)]\n    return [round(v) for v in filtered] if round_result else filtered",
    "response": '"""Filter a list of numbers within a range.\n\nArgs:\n    values: List of numbers\n    low: Lower bound\n    high: Upper bound\n    inclusive: Include boundaries if True\n    round_result: Round numbers if True\n\nReturns:\n    Filtered list of numbers\n"""'
},
{
    "instruction": "def mask_email(email, mask_char='*', keep_domain=True, min_visible=2):\n    local, domain = email.split('@')\n    visible = local[:min_visible]\n    masked = visible + mask_char * (len(local) - min_visible)\n    return f'{masked}@{domain}' if keep_domain else masked",
    "response": '"""Mask the local part of an email address.\n\nArgs:\n    email: Email address\n    mask_char: Character used to mask\n    keep_domain: Whether to retain domain part\n    min_visible: Number of visible characters in local part\n\nReturns:\n    Masked email string\n"""'
},
{
    "instruction": "def concat_fields(a, b, c, d, separator=', '):\n    return separator.join([a, b, c, d])",
    "response": '"""Concatenate four strings with a separator.\n\nArgs:\n    a: First string\n    b: Second string\n    c: Third string\n    d: Fourth string\n    separator: Delimiter to use\n\nReturns:\n    Concatenated string\n"""'
},
{
    "instruction": "def normalize_vector(x, y, z, precision=2):\n    norm = (x**2 + y**2 + z**2) ** 0.5\n    return [round(x/norm, precision), round(y/norm, precision), round(z/norm, precision)]",
    "response": '"""Normalize a 3D vector.\n\nArgs:\n    x: X component\n    y: Y component\n    z: Z component\n    precision: Decimal precision for rounding\n\nReturns:\n    Normalized vector as list\n"""'
},
{
    "instruction": "def rescale(value, old_min, old_max, new_min, new_max):\n    scale = (new_max - new_min) / (old_max - old_min)\n    return new_min + (value - old_min) * scale",
    "response": '"""Rescale a number from one range to another.\n\nArgs:\n    value: Input number\n    old_min: Original minimum\n    old_max: Original maximum\n    new_min: Target minimum\n    new_max: Target maximum\n\nReturns:\n    Rescaled value\n"""'
},
{
    "instruction": "def crop_image(img, x, y, width, height):\n    return img[y:y+height, x:x+width]",
    "response": '"""Crop an image array to a rectangle.\n\nArgs:\n    img: Image array\n    x: Top-left x coordinate\n    y: Top-left y coordinate\n    width: Width of crop\n    height: Height of crop\n\nReturns:\n    Cropped image array\n"""'
},
{
    "instruction": "def log_event(user_id, action, timestamp, metadata=None, source='web'):\n    return {\n        'user': user_id,\n        'action': action,\n        'time': timestamp,\n        'meta': metadata or {},\n        'source': source\n    }",
    "response": '"""Create a structured event log entry.\n\nArgs:\n    user_id: Identifier for the user\n    action: Description of the action\n    timestamp: Time of the event\n    metadata: Optional metadata dictionary\n    source: Origin of the event\n\nReturns:\n    Dictionary representing the event\n"""'
},
{
    "instruction": "def paginate_items(items, page, per_page, include_total=False, label='results'):\n    start = (page - 1) * per_page\n    end = start + per_page\n    output = {label: items[start:end]}\n    if include_total:\n        output['total'] = len(items)\n    return output",
    "response": '"""Paginate a list of items.\n\nArgs:\n    items: Full list of data\n    page: Page number (1-indexed)\n    per_page: Items per page\n    include_total: Whether to include total count\n    label: Key name for result list\n\nReturns:\n    Dictionary with paginated data\n"""'
},
{
    "instruction": "def format_currency(amount, symbol='$', decimals=2, separator=',', trailing_zeros=True):\n    formatted = f\"{amount:,.{decimals}f}\" if trailing_zeros else f\"{amount:,.{decimals}g}\"\n    return f\"{symbol}{formatted.replace(',', separator)}\"",
    "response": '"""Format a number as currency.\n\nArgs:\n    amount: Numeric value\n    symbol: Currency symbol\n    decimals: Decimal precision\n    separator: Thousands separator\n    trailing_zeros: Whether to include trailing zeros\n\nReturns:\n    Formatted currency string\n"""'
},
{
    "instruction": "def validate_range(value, min_val, max_val, step=1, inclusive=True):\n    if not inclusive and (value <= min_val or value >= max_val):\n        return False\n    if inclusive and (value < min_val or value > max_val):\n        return False\n    return (value - min_val) % step == 0",
    "response": '"""Check if a value falls within a specified range and step.\n\nArgs:\n    value: Value to check\n    min_val: Minimum allowed\n    max_val: Maximum allowed\n    step: Step interval\n    inclusive: Include boundaries if True\n\nReturns:\n    True if valid, else False\n"""'
},
{
    "instruction": "def safe_cast(value, target_type, fallback=None, strict=False):\n    try:\n        return target_type(value)\n    except Exception:\n        if strict:\n            raise\n        return fallback",
    "response": '"""Attempt to cast a value to a specific type safely.\n\nArgs:\n    value: Value to cast\n    target_type: Target type or callable\n    fallback: Value to return if cast fails\n    strict: Raise exception if True\n\nReturns:\n    Cast value or fallback\n"""'
},
{
    "instruction": "def serialize_user(id, name, active, created_at, include_meta=False):\n    user = {'id': id, 'name': name, 'active': active, 'created': created_at.isoformat()}\n    if include_meta:\n        user['meta'] = {'length': len(name)}\n    return user",
    "response": '"""Serialize user data to a dictionary.\n\nArgs:\n    id: User identifier\n    name: Full name\n    active: Active status\n    created_at: Creation datetime\n    include_meta: Whether to include metadata\n\nReturns:\n    Serialized user dictionary\n"""'
},
{
    "instruction": "def batch_prefix(items, prefix, suffix='', max_items=10, pad_width=3):\n    return [f\"{prefix}{str(i).zfill(pad_width)}{suffix}\" for i in range(min(len(items), max_items))]",
    "response": '"""Generate a list of prefixed strings from a sequence.\n\nArgs:\n    items: List of source items\n    prefix: Prefix for each string\n    suffix: Optional suffix\n    max_items: Maximum number of results\n    pad_width: Width for zero padding\n\nReturns:\n    List of formatted strings\n"""'
},
{
    "instruction": "def get_grade(score):\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    elif score >= 70:\n        return 'C'\n    elif score >= 60:\n        return 'D'\n    else:\n        return 'F'",
    "response": '"""Convert numeric score to letter grade.\n\nArgs:\n    score: Numeric grade from 0 to 100\n\nReturns:\n    Letter grade A–F\n"""'
},
{
    "instruction": "def get_file_extension(filename):\n    if '.' not in filename:\n        return None\n    parts = filename.rsplit('.', 1)\n    if len(parts) != 2:\n        return None\n    return parts[1].lower()",
    "response": '"""Extract the file extension from a filename.\n\nArgs:\n    filename: Name of the file\n\nReturns:\n    File extension in lowercase, or None if not found\n"""'
},
{
    "instruction": "def compute_fee(amount, vip=False):\n    if amount <= 0:\n        return 0\n    fee = 0.05 * amount\n    if vip:\n        return fee * 0.5\n    return fee",
    "response": '"""Calculate a transaction fee.\n\nArgs:\n    amount: Transaction amount\n    vip: Whether the user is a VIP\n\nReturns:\n    Fee amount, discounted for VIPs\n"""'
},
{
    "instruction": "def validate_password(password):\n    if not password:\n        return 'empty'\n    if len(password) < 8:\n        return 'too short'\n    if password.isnumeric():\n        return 'no letters'\n    return 'valid'",
    "response": '"""Validate a password string.\n\nArgs:\n    password: Password to validate\n\nReturns:\n    Validation status string\n"""'
},
{
    "instruction": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
    "response": '"""Perform safe division.\n\nArgs:\n    a: Numerator\n    b: Denominator\n\nReturns:\n    Quotient, or None if dividing by zero\n"""'
},
{
    "instruction": "def parse_int(value):\n    try:\n        return int(value)\n    except (ValueError, TypeError):\n        return None",
    "response": '"""Convert a value to integer.\n\nArgs:\n    value: Value to convert\n\nReturns:\n    Integer value or None if conversion fails\n"""'
},
{
    "instruction": "def get_status_label(code):\n    if code in (200, 201):\n        return 'success'\n    elif code in (400, 404):\n        return 'client error'\n    elif code in (500, 503):\n        return 'server error'\n    return 'unknown'",
    "response": '"""Map HTTP status code to category.\n\nArgs:\n    code: HTTP status code\n\nReturns:\n    Status label as string\n"""'
},
{
    "instruction": "def categorize_number(n):\n    if n == 0:\n        return 'zero'\n    if n > 0:\n        if n % 2 == 0:\n            return 'positive even'\n        return 'positive odd'\n    else:\n        return 'negative'",
    "response": '"""Categorize a number based on sign and parity.\n\nArgs:\n    n: Number to categorize\n\nReturns:\n    String label describing the number\n"""'
},
{
    "instruction": "def check_user_access(role, active, expired):\n    if not active or expired:\n        return 'denied'\n    if role == 'admin':\n        return 'full'\n    elif role == 'user':\n        return 'limited'\n    return 'none'",
    "response": '"""Determine user access level.\n\nArgs:\n    role: Role of the user\n    active: Whether account is active\n    expired: Whether account is expired\n\nReturns:\n    Access level as string\n"""'
},
{
    "instruction": "def get_triangle_type(a, b, c):\n    if a + b <= c or a + c <= b or b + c <= a:\n        return 'invalid'\n    if a == b == c:\n        return 'equilateral'\n    elif a == b or b == c or a == c:\n        return 'isosceles'\n    return 'scalene'",
    "response": '"""Classify a triangle by side lengths.\n\nArgs:\n    a: Side 1\n    b: Side 2\n    c: Side 3\n\nReturns:\n    Triangle type as string\n"""'
},
{
    "instruction": "def pick_color(code):\n    if code == 1:\n        return 'red'\n    elif code == 2:\n        return 'green'\n    elif code == 3:\n        return 'blue'\n    return 'unknown'",
    "response": '"""Return a color string based on code.\n\nArgs:\n    code: Integer color code\n\nReturns:\n    Color name as string\n"""'
},
{
    "instruction": "def is_valid_email(email):\n    if not isinstance(email, str):\n        return False\n    if '@' not in email or '.' not in email:\n        return False\n    return True",
    "response": '"""Check if an email string is valid.\n\nArgs:\n    email: Email address to validate\n\nReturns:\n    True if valid, False otherwise\n"""'
},
{
    "instruction": "def safe_get(lst, index):\n    if not isinstance(lst, list):\n        return None\n    if index < 0 or index >= len(lst):\n        return None\n    return lst[index]",
    "response": '"""Safely retrieve an element from a list.\n\nArgs:\n    lst: Input list\n    index: Index to retrieve\n\nReturns:\n    List element or None if out of bounds\n"""'
},
{
    "instruction": "def convert_case(text, to_upper=True):\n    if not isinstance(text, str):\n        return text\n    return text.upper() if to_upper else text.lower()",
    "response": '"""Convert string to upper or lower case.\n\nArgs:\n    text: Input string\n    to_upper: If True, convert to uppercase\n\nReturns:\n    Converted string or original input if not a string\n"""'
},
{
    "instruction": "def classify_temp(temp):\n    if temp < 0:\n        return 'freezing'\n    elif temp < 15:\n        return 'cold'\n    elif temp < 25:\n        return 'moderate'\n    else:\n        return 'hot'",
    "response": '"""Classify temperature level.\n\nArgs:\n    temp: Temperature in Celsius\n\nReturns:\n    Temperature category as string\n"""'
},
{
    "instruction": "def choose_plan(income):\n    if income < 20000:\n        return 'basic'\n    if income < 50000:\n        return 'standard'\n    if income < 100000:\n        return 'premium'\n    return 'elite'",
    "response": '"""Select a plan based on income bracket.\n\nArgs:\n    income: Annual income amount\n\nReturns:\n    Plan name as string\n"""'
},
{
    "instruction": "def process_flag(flag):\n    if flag is None:\n        return 'missing'\n    if flag is True:\n        return 'enabled'\n    if flag is False:\n        return 'disabled'\n    return 'invalid'",
    "response": '"""Interpret a boolean flag.\n\nArgs:\n    flag: Boolean or None\n\nReturns:\n    String interpretation of the flag\n"""'
},
{
    "instruction": "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total",
    "response": '"""Compute the sum of a list of numbers.\n\nArgs:\n    numbers: List of numeric values\n\nReturns:\n    Sum of the numbers\n"""'
},
{
    "instruction": "def count_occurrences(items, target):\n    count = 0\n    for item in items:\n        if item == target:\n            count += 1\n    return count",
    "response": '"""Count how many times a target appears in a list.\n\nArgs:\n    items: List of elements\n    target: Value to count\n\nReturns:\n    Number of times the target appears\n"""'
},
{
    "instruction": "def has_duplicates(seq):\n    seen = set()\n    for item in seq:\n        if item in seen:\n            return True\n        seen.add(item)\n    return False",
    "response": '"""Check if a sequence contains duplicate values.\n\nArgs:\n    seq: List or iterable of hashable elements\n\nReturns:\n    True if any duplicates exist, else False\n"""'
},
{
    "instruction": "def flatten(matrix):\n    flat = []\n    for row in matrix:\n        for item in row:\n            flat.append(item)\n    return flat",
    "response": '"""Flatten a 2D list into a 1D list.\n\nArgs:\n    matrix: List of lists\n\nReturns:\n    Flattened list\n"""'
},
{
    "instruction": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "response": '"""Compute factorial of a number recursively.\n\nArgs:\n    n: Non-negative integer\n\nReturns:\n    Factorial of n\n"""'
},
{
    "instruction": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "response": '"""Compute nth Fibonacci number recursively.\n\nArgs:\n    n: Index in Fibonacci sequence\n\nReturns:\n    Fibonacci number at position n\n"""'
},
{
    "instruction": "def sum_even(numbers):\n    total = 0\n    for n in numbers:\n        if n % 2 == 0:\n            total += n\n    return total",
    "response": '"""Sum the even numbers in a list.\n\nArgs:\n    numbers: List of integers\n\nReturns:\n    Sum of even numbers\n"""'
},
{
    "instruction": "def find_max(values):\n    if not values:\n        return None\n    max_val = values[0]\n    for val in values:\n        if val > max_val:\n            max_val = val\n    return max_val",
    "response": '"""Find the maximum value in a list.\n\nArgs:\n    values: List of comparable elements\n\nReturns:\n    Maximum value or None if list is empty\n"""'
},
{
    "instruction": "def filter_long_words(words, length):\n    result = []\n    for word in words:\n        if len(word) > length:\n            result.append(word)\n    return result",
    "response": '"""Filter words longer than a given length.\n\nArgs:\n    words: List of strings\n    length: Minimum length threshold\n\nReturns:\n    List of words longer than threshold\n"""'
},
{
    "instruction": "def remove_negatives(nums):\n    result = []\n    for n in nums:\n        if n >= 0:\n            result.append(n)\n    return result",
    "response": '"""Remove negative numbers from a list.\n\nArgs:\n    nums: List of numbers\n\nReturns:\n    List of non-negative numbers\n"""'
},
{
    "instruction": "def all_upper(strings):\n    result = []\n    for s in strings:\n        result.append(s.upper())\n    return result",
    "response": '"""Convert all strings in a list to uppercase.\n\nArgs:\n    strings: List of strings\n\nReturns:\n    List of uppercase strings\n"""'
},
{
    "instruction": "def product(numbers):\n    result = 1\n    for n in numbers:\n        result *= n\n    return result",
    "response": '"""Calculate the product of a list of numbers.\n\nArgs:\n    numbers: List of numeric values\n\nReturns:\n    Product of all elements\n"""'
},
{
    "instruction": "def zip_lists(a, b):\n    result = []\n    for i in range(min(len(a), len(b))):\n        result.append((a[i], b[i]))\n    return result",
    "response": '"""Pair elements from two lists.\n\nArgs:\n    a: First list\n    b: Second list\n\nReturns:\n    List of tuple pairs from a and b\n"""'
},
{
    "instruction": "def recursive_sum(lst):\n    if not lst:\n        return 0\n    return lst[0] + recursive_sum(lst[1:])",
    "response": '"""Recursively sum a list of numbers.\n\nArgs:\n    lst: List of numeric values\n\nReturns:\n    Total sum of list\n"""'
},
{
    "instruction": "def reverse_list_in_place(lst):\n    n = len(lst)\n    for i in range(n // 2):\n        lst[i], lst[n - 1 - i] = lst[n - 1 - i], lst[i]",
    "response": '"""Reverse a list in place.\n\nArgs:\n    lst: List to reverse\n\nReturns:\n    None. The input list is modified in place.\n"""'
},
{
    "instruction": "def unique_values(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result",
    "response": '"""Return list of unique elements in original order.\n\nArgs:\n    items: List of hashable values\n\nReturns:\n    List without duplicates\n"""'
},
{
    "instruction": "def cumulative_sum(nums):\n    result = []\n    total = 0\n    for n in nums:\n        total += n\n        result.append(total)\n    return result",
    "response": '"""Generate cumulative sums from a list of numbers.\n\nArgs:\n    nums: List of numbers\n\nReturns:\n    List of running totals\n"""'
},
{
    "instruction": "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
    "response": '"""Check if a string is a palindrome.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if palindrome, else False\n"""'
},
{
    "instruction": "def count_characters(text):\n    counts = {}\n    for char in text:\n        if char in counts:\n            counts[char] += 1\n        else:\n            counts[char] = 1\n    return counts",
    "response": '"""Count character frequencies in a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    Dictionary of character counts\n"""'
},
{
    "instruction": "def find_common_elements(a, b):\n    result = []\n    for item in a:\n        if item in b:\n            result.append(item)\n    return result",
    "response": '"""Find common elements in two lists.\n\nArgs:\n    a: First list\n    b: Second list\n\nReturns:\n    List of elements present in both a and b\n"""'
},
{
    "instruction": "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
    "response": '"""Check if a string is a palindrome.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if the string is a palindrome, False otherwise\n"""'
},
{
    "instruction": "def count_vowels(text):\n    return sum(1 for char in text.lower() if char in 'aeiou')",
    "response": '"""Count vowels in a string.\n\nArgs:\n    text: Input string\n\nReturns:\n    Number of vowels as integer\n"""'
},
{
    "instruction": "def flatten_list(nested):\n    return [item for sublist in nested for item in sublist]",
    "response": '"""Flatten a list of lists into a single list.\n\nArgs:\n    nested: List of lists\n\nReturns:\n    Flattened list of items\n"""'
},
{
    "instruction": "def remove_duplicates(seq):\n    seen = set()\n    return [x for x in seq if not (x in seen or seen.add(x))]",
    "response": '"""Remove duplicates while preserving order.\n\nArgs:\n    seq: Input list\n\nReturns:\n    Deduplicated list\n"""'
},
{
    "instruction": "def validate_ip(ip):\n    parts = ip.split('.')\n    if len(parts) != 4:\n        return False\n    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)",
    "response": '"""Validate if a string is a valid IPv4 address.\n\nArgs:\n    ip: IP address string\n\nReturns:\n    True if valid, else False\n"""'
},
{
    "instruction": "def get_env_variable(key, default=None):\n    import os\n    return os.getenv(key, default)",
    "response": '"""Get environment variable or default.\n\nArgs:\n    key: Variable name\n    default: Value to return if key is not set\n\nReturns:\n    Value of the environment variable or default\n"""'
},
{
    "instruction": "def kebab_to_snake(s):\n    return s.replace('-', '_')",
    "response": '"""Convert kebab-case string to snake_case.\n\nArgs:\n    s: Kebab-case string\n\nReturns:\n    Snake_case string\n"""'
},
{
    "instruction": "def to_camel_case(s):\n    parts = s.split('_')\n    return parts[0] + ''.join(p.capitalize() for p in parts[1:])",
    "response": '"""Convert snake_case to camelCase.\n\nArgs:\n    s: Snake_case string\n\nReturns:\n    camelCase version of the string\n"""'
},
{
    "instruction": "def safe_int(value, default=0):\n    try:\n        return int(value)\n    except (ValueError, TypeError):\n        return default",
    "response": '"""Convert value to int with fallback.\n\nArgs:\n    value: Value to convert\n    default: Fallback if conversion fails\n\nReturns:\n    Integer value or fallback\n"""'
},
{
    "instruction": "def pluralize(word, count):\n    return word if count == 1 else word + 's'",
    "response": '"""Return singular or plural form of a word.\n\nArgs:\n    word: Base word\n    count: Quantity of the word\n\nReturns:\n    Word in correct singular/plural form\n"""'
},
{
    "instruction": "def is_palindrome_ignore_case(s):\n    return s.lower() == s[::-1].lower()",
    "response": "'''Check if a string is a palindrome, ignoring case.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if palindrome, else False\n'''"
},
{
    "instruction": "def clamp(value, minimum, maximum):\n    return max(minimum, min(value, maximum))",
    "response": "'''Clamp a value between a minimum and maximum.\n\nArgs:\n    value: The input number\n    minimum: Lower bound\n    maximum: Upper bound\n\nReturns:\n    Clamped value\n'''"
},
{
    "instruction": "def reverse_dict(d):\n    return {v: k for k, v in d.items()}",
    "response": "'''Reverse keys and values in a dictionary.\n\nArgs:\n    d: Dictionary to reverse\n\nReturns:\n    Dictionary with values as keys\n'''"
},
{
    "instruction": "def safe_divide(a, b):\n    return a / b if b != 0 else None",
    "response": "'''Divide two numbers safely.\n\nArgs:\n    a: Numerator\n    b: Denominator\n\nReturns:\n    Result of division or None if divide by zero\n'''"
},
{
    "instruction": "def has_duplicate_items(seq):\n    return len(seq) != len(set(seq))",
    "response": "'''Check if a sequence has duplicate items.\n\nArgs:\n    seq: Input list or sequence\n\nReturns:\n    True if duplicates exist\n'''"
},
{
    "instruction": "def list_to_csv(values):\n    return ','.join(map(str, values))",
    "response": "'''Convert a list of values to a CSV string.\n\nArgs:\n    values: List of items\n\nReturns:\n    Comma-separated string\n'''"
},
{
    "instruction": "def count_true(flags):\n    return sum(1 for f in flags if f)",
    "response": "'''Count the number of True values in a list.\n\nArgs:\n    flags: List of boolean values\n\nReturns:\n    Number of True entries\n'''"
},
{
    "instruction": "def is_hex(s):\n    try:\n        int(s, 16)\n        return True\n    except ValueError:\n        return False",
    "response": "'''Check if a string is a valid hexadecimal number.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if hex, else False\n'''"
},
{
    "instruction": "def xor_bool(a, b):\n    return bool(a) ^ bool(b)",
    "response": "'''Perform exclusive OR on two boolean values.\n\nArgs:\n    a: First boolean\n    b: Second boolean\n\nReturns:\n    True if exactly one is True\n'''"
},
{
    "instruction": "def is_subpath(path, base):\n    return path.startswith(base)",
    "response": "'''Check if a path starts with a base path.\n\nArgs:\n    path: Full path\n    base: Base path to check\n\nReturns:\n    True if path is within base\n'''"
},
{
    "instruction": "def has_emoji(text):\n    return any(ord(c) > 10000 for c in text)",
    "response": "'''Detect if a string contains emoji-like characters.\n\nArgs:\n    text: Input string\n\nReturns:\n    True if emoji present, else False\n'''"
},
{
    "instruction": "def strip_shell_comments(line):\n    return line.split('#', 1)[0].rstrip()",
    "response": "'''Remove shell-style comments from a line.\n\nArgs:\n    line: A single line of text\n\nReturns:\n    Line with comments stripped\n'''"
},
{
    "instruction": "def get_file_extension(filename):\n    return filename.rsplit('.', 1)[-1] if '.' in filename else ''",
    "response": "'''Extract the file extension from a filename.\n\nArgs:\n    filename: Name of the file\n\nReturns:\n    File extension or empty string\n'''"
},
{
    "instruction": "def is_pangram(sentence):\n    return set('abcdefghijklmnopqrstuvwxyz').issubset(sentence.lower())",
    "response": "'''Check if a sentence is a pangram.\n\nArgs:\n    sentence: Input sentence\n\nReturns:\n    True if all letters of alphabet are present\n'''"
},
{
    "instruction": "def count_leading_spaces(line):\n    return len(line) - len(line.lstrip(' '))",
    "response": "'''Count number of leading spaces in a string.\n\nArgs:\n    line: A single line of text\n\nReturns:\n    Number of leading space characters\n'''"
},
{
    "instruction": "def percent_diff(a, b):\n    return 0 if a == 0 else abs(a - b) / abs(a) * 100",
    "response": "'''Calculate percentage difference between two values.\n\nArgs:\n    a: Original value\n    b: New value\n\nReturns:\n    Percentage difference\n'''"
},
{
    "instruction": "def is_valid_ipv4(s):\n    parts = s.split('.')\n    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)",
    "response": "'''Validate a string as a basic IPv4 address.\n\nArgs:\n    s: IP address string\n\nReturns:\n    True if valid IPv4 format\n'''"
},
{
    "instruction": "def redact_digits(s):\n    return ''.join('X' if c.isdigit() else c for c in s)",
    "response": "'''Redact digits from a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    String with digits replaced by 'X'\n'''"
},
{
    "instruction": "def longest_common_prefix(strings):\n    return os.path.commonprefix(strings) if strings else ''",
    "response": "'''Find the longest common prefix among strings.\n\nArgs:\n    strings: List of strings\n\nReturns:\n    Longest shared prefix\n'''"
},
{
    "instruction": "def parse_flags(flag_string):\n    return [f.strip() for f in flag_string.split(',') if f.strip()]",
    "response": "'''Split a comma-separated flag string into a list.\n\nArgs:\n    flag_string: String of flags\n\nReturns:\n    List of non-empty trimmed flags\n'''"
},
{
    "instruction": "def has_unusual_whitespace(s):\n    return any(ord(c) in {0x200B, 0x200C, 0x200D} for c in s)",
    "response": "'''Check for presence of zero-width or unusual whitespace characters.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if unusual whitespace is present\n'''"
},
{
    "instruction": "def toggle_trailing_slash(path):\n    return path[:-1] if path.endswith('/') else path + '/'",
    "response": "'''Toggle the presence of a trailing slash in a path.\n\nArgs:\n    path: Filesystem or URL path\n\nReturns:\n    Path with toggled slash\n'''"
},
{
    "instruction": "def is_single_bit_set(n):\n    return n > 0 and (n & (n - 1)) == 0",
    "response": "'''Check if an integer has only a single bit set.\n\nArgs:\n    n: Integer to check\n\nReturns:\n    True if only one bit is set\n'''"
},
{
    "instruction": "def extract_digits(s):\n    return ''.join(c for c in s if c.isdigit())",
    "response": "'''Extract only the digit characters from a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    String of numeric characters\n'''"
},
{
    "instruction": "def is_weird_case(s):\n    return s != s.lower() and s != s.upper() and not s.istitle()",
    "response": "'''Determine if a string uses inconsistent casing.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if casing appears irregular\n'''"
},
{
    "instruction": "def decode_rot13(s):\n    return s.translate(str.maketrans(\n        'ABCDEFGHIJKLMabcdefghijklmNOPQRSTUVWXYZnopqrstuvwxyz',\n        'NOPQRSTUVWXYZnopqrstuvwxyzABCDEFGHIJKLMabcdefghijklm'\n    ))",
    "response": "'''Decode a ROT13-encoded string.\n\nArgs:\n    s: ROT13 encoded text\n\nReturns:\n    Decoded string\n'''"
},
{
    "instruction": "def join_nonempty_lines(lines):\n    return '\\n'.join(line for line in lines if line.strip())",
    "response": "'''Join only non-empty lines with newlines.\n\nArgs:\n    lines: List of strings\n\nReturns:\n    String of joined non-blank lines\n'''"
},
{
    "instruction": "def count_unicode_categories(s, prefix):\n    import unicodedata\n    return sum(unicodedata.category(c).startswith(prefix) for c in s)",
    "response": "'''Count characters in a string that match a Unicode category prefix.\n\nArgs:\n    s: Input string\n    prefix: Unicode category prefix (e.g., 'L', 'P')\n\nReturns:\n    Number of matching characters\n'''"
},
{
    "instruction": "def quote_wrap(text, quote_char='\"'):\n    return f'{quote_char}{text}{quote_char}'",
    "response": "'''Wrap a string in matching quote characters.\n\nArgs:\n    text: The string to wrap\n    quote_char: The character to use for wrapping (default is \")\n\nReturns:\n    Quoted string\n'''"
},
{
    "instruction": "def is_sequential_ascii(chars):\n    return all(ord(b) - ord(a) == 1 for a, b in zip(chars, chars[1:]))",
    "response": "'''Check if characters form a sequential ASCII range.\n\nArgs:\n    chars: String of characters\n\nReturns:\n    True if characters are sequential ASCII\n'''"
},
{
    "instruction": "def strip_non_ascii(s):\n    return ''.join(c for c in s if ord(c) < 128)",
    "response": "'''Remove all non-ASCII characters from a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    ASCII-only string\n'''"
},
{
    "instruction": "def find_all_caps_words(text):\n    return [w for w in text.split() if w.isupper()]",
    "response": "'''Extract all fully capitalized words from text.\n\nArgs:\n    text: Input string\n\nReturns:\n    List of uppercase words\n'''"
},
{
    "instruction": "def remove_duplicate_lines(lines):\n    seen = set()\n    return [l for l in lines if not (l in seen or seen.add(l))]",
    "response": "'''Remove duplicate lines while preserving order.\n\nArgs:\n    lines: List of strings\n\nReturns:\n    List with duplicates removed\n'''"
},
{
    "instruction": "def wrap_in_markdown_code(s, lang=''):\n    return f'```{lang}\\n{s}\\n```'",
    "response": "'''Wrap text in a fenced Markdown code block.\n\nArgs:\n    s: Code or text to wrap\n    lang: Optional language tag\n\nReturns:\n    Markdown-formatted string\n'''"
},
{
    "instruction": "def has_mixed_indentation(code):\n    return '\\t' in code and '    ' in code",
    "response": "'''Check if code string contains mixed tabs and spaces.\n\nArgs:\n    code: Code snippet as string\n\nReturns:\n    True if both tabs and spaces are used\n'''"
},
{
    "instruction": "def dedent_lines(lines):\n    import textwrap\n    return textwrap.dedent('\\n'.join(lines)).splitlines()",
    "response": "'''Dedent a list of lines to remove common indentation.\n\nArgs:\n    lines: List of indented strings\n\nReturns:\n    List of dedented lines\n'''"
},
{
    "instruction": "def truncate_middle(s, max_length):\n    if len(s) <= max_length:\n        return s\n    half = (max_length - 3) // 2\n    return s[:half] + '...' + s[-half:]",
    "response": "'''Truncate a string in the middle to fit a max length.\n\nArgs:\n    s: Input string\n    max_length: Maximum allowed length\n\nReturns:\n    Truncated string with ellipsis in the middle\n'''"
},
{
    "instruction": "def has_balanced_quotes(s):\n    return s.count('\"') % 2 == 0 and s.count(\"'\") % 2 == 0",
    "response": "'''Check if a string has balanced single and double quotes.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if all quotes are balanced\n'''"
},
{
    "instruction": "def pad_left(s, width, char='0'):\n    return s.rjust(width, char)",
    "response": "'''Left-pad a string to a given width using a specific character.\n\nArgs:\n    s: Input string\n    width: Desired width\n    char: Padding character (default is '0')\n\nReturns:\n    Left-padded string\n'''"
},
{
    "instruction": "def is_valid_python_identifier(s):\n    return s.isidentifier() and not s.isnumeric()",
    "response": "'''Determine if a string is a valid Python identifier.\n\nArgs:\n    s: String to validate\n\nReturns:\n    True if it's a legal identifier\n'''"
},
{
    "instruction": "def has_invisible_control_chars(s):\n    return any(0 < ord(c) < 32 and c not in ('\\n', '\\t') for c in s)",
    "response": "'''Detect control characters that are not newline or tab.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if non-printable controls are present\n'''"
},
{
    "instruction": "def split_at_last_colon(s):\n    return s.rsplit(':', 1) if ':' in s else [s]",
    "response": "'''Split a string at the last colon.\n\nArgs:\n    s: Colon-delimited string\n\nReturns:\n    List with at most two parts\n'''"
},
{
    "instruction": "def strip_ansi_codes(text):\n    import re\n    return re.sub(r'\\x1B\\[[0-?]*[ -/]*[@-~]', '', text)",
    "response": "'''Remove ANSI escape codes from terminal text.\n\nArgs:\n    text: String with potential ANSI codes\n\nReturns:\n    Cleaned string without color/control sequences\n'''"
},
{
    "instruction": "def parse_shell_vars(s):\n    return dict(pair.split('=') for pair in s.strip().split() if '=' in pair)",
    "response": "'''Parse space-separated shell-style KEY=VALUE pairs.\n\nArgs:\n    s: Input string of assignments\n\nReturns:\n    Dictionary of parsed variables\n'''"
},
{
    "instruction": "def normalize_windows_path(path):\n    return path.replace('/', '\\\\')",
    "response": "'''Convert forward slashes to backslashes for Windows paths.\n\nArgs:\n    path: File path string\n\nReturns:\n    Windows-compatible path\n'''"
},
{
    "instruction": "def contains_bidi_chars(s):\n    return any(0x202A <= ord(c) <= 0x202E for c in s)",
    "response": "'''Check for presence of bidirectional override characters.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if Bidi control characters found\n'''"
},
{
    "instruction": "def count_trailing_newlines(s):\n    return len(s) - len(s.rstrip('\\n'))",
    "response": "'''Count number of trailing newline characters in a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    Number of '\\n' at end of string\n'''"
},
{
    "instruction": "def shorten_hostname(fqdn):\n    return fqdn.split('.')[0]",
    "response": "'''Extract short hostname from fully qualified domain name.\n\nArgs:\n    fqdn: Full domain string\n\nReturns:\n    Hostname prefix before first dot\n'''"
},
{
    "instruction": "def looks_like_hex_token(s):\n    return len(s) >= 16 and all(c in '0123456789abcdefABCDEF' for c in s)",
    "response": "'''Heuristically detect if a string resembles a hex token.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if long hex string\n'''"
},
{
    "instruction": "def extract_after_last_dot(s):\n    return s.rsplit('.', 1)[-1] if '.' in s else ''",
    "response": "'''Get the text after the last dot in a string.\n\nArgs:\n    s: Dot-delimited string\n\nReturns:\n    Final segment or empty string\n'''"
},
{
    "instruction": "def is_windows_drive_path(p):\n    return len(p) >= 2 and p[1] == ':' and p[0].isalpha()",
    "response": "'''Check if a string looks like a Windows drive path.\n\nArgs:\n    p: Input path string\n\nReturns:\n    True if format matches drive letter colon\n'''"
},
{
    "instruction": "def truncate_lines(lines, limit):\n    return lines[:limit] + ['...'] if len(lines) > limit else lines",
    "response": "'''Limit a list of lines and append ellipsis if truncated.\n\nArgs:\n    lines: List of strings\n    limit: Max number of lines to keep\n\nReturns:\n    Possibly-truncated list with optional ellipsis\n'''"
},
{
    "instruction": "def extract_inline_comment(line):\n    return line.split('#', 1)[1].strip() if '#' in line else ''",
    "response": "'''Extract comment content from a Python-style code line.\n\nArgs:\n    line: Code line with optional inline comment\n\nReturns:\n    Comment text or empty string\n'''"
},
{
    "instruction": "def remove_duplicate_keys(pairs):\n    seen = set()\n    result = []\n    for k, v in pairs:\n        if k not in seen:\n            seen.add(k)\n            result.append((k, v))\n    return result",
    "response": "'''Remove later duplicates from a list of key-value pairs.\n\nArgs:\n    pairs: List of (key, value) tuples\n\nReturns:\n    List without duplicate keys\n'''"
},
{
    "instruction": "def ends_with_number(s):\n    return s.rstrip().split()[-1].isdigit() if s.strip() else False",
    "response": "'''Check if the last word in a string is numeric.\n\nArgs:\n    s: Input sentence or string\n\nReturns:\n    True if last token is a number\n'''"
},
{
    "instruction": "def decode_url_encoded(s):\n    from urllib.parse import unquote\n    return unquote(s)",
    "response": "'''Decode percent-encoded characters in a URL.\n\nArgs:\n    s: URL-encoded string\n\nReturns:\n    Decoded text string\n'''"
},
{
    "instruction": "def is_ascii_punctuation_only(s):\n    return all(c in '.,:;!?\"\'-()[]{}' for c in s)",
    "response": "'''Check if a string contains only basic ASCII punctuation.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if only punctuation characters\n'''"
},
{
    "instruction": "def escape_double_quotes(s):\n    return s.replace('\"', '\\\\\"')",
    "response": "'''Escape all double quotes in a string.\n\nArgs:\n    s: Input string\n\nReturns:\n    String with double quotes escaped\n'''"
},
{
    "instruction": "def unix_timestamp_to_iso(ts):\n    from datetime import datetime\n    return datetime.utcfromtimestamp(ts).isoformat() + 'Z'",
    "response": "'''Convert a Unix timestamp to ISO 8601 UTC format.\n\nArgs:\n    ts: Timestamp as integer or float\n\nReturns:\n    ISO-formatted datetime string\n'''"
},
{
    "instruction": "def has_printable_non_ascii(s):\n    return any(ord(c) > 127 and c.isprintable() for c in s)",
    "response": "'''Detect if a string contains printable non-ASCII characters.\n\nArgs:\n    s: Input string\n\nReturns:\n    True if printable Unicode beyond ASCII\n'''"
},
{
    "instruction": "def is_policy_lapsed(last_payment_date, today):\n    return (today - last_payment_date).days > 30",
    "response": "'''Check if a policy has lapsed based on payment history.\n\nArgs:\n    last_payment_date: Date of last payment\n    today: Current date\n\nReturns:\n    True if lapsed, else False\n'''"
},
{
    "instruction": "def earned_premium(written_premium, term_days, earned_days):\n    return written_premium * earned_days / term_days",
    "response": "'''Calculate earned premium using pro rata method.\n\nArgs:\n    written_premium: Total written premium\n    term_days: Policy term in days\n    earned_days: Days elapsed in term\n\nReturns:\n    Earned premium amount\n'''"
},
{
    "instruction": "def is_loss_ratio_high(claims, premium):\n    return claims / premium > 0.7 if premium else False",
    "response": "'''Evaluate if loss ratio exceeds threshold.\n\nArgs:\n    claims: Total incurred claims\n    premium: Earned premium\n\nReturns:\n    True if ratio > 0.7\n'''"
},
{
    "instruction": "def expected_claim_count(exposure, frequency):\n    return exposure * frequency",
    "response": "'''Estimate expected claim count from exposure and frequency.\n\nArgs:\n    exposure: Risk exposure (e.g., car-years)\n    frequency: Claims per unit exposure\n\nReturns:\n    Expected number of claims\n'''"
},
{
    "instruction": "def average_severity(total_loss, count):\n    return total_loss / count if count else 0.0",
    "response": "'''Compute average claim severity.\n\nArgs:\n    total_loss: Sum of claim amounts\n    count: Number of claims\n\nReturns:\n    Average severity value\n'''"
},
{
    "instruction": "def credibility_weight(n, threshold=1000):\n    return min(n / threshold, 1.0)",
    "response": "'''Calculate limited credibility weight.\n\nArgs:\n    n: Number of claims or observations\n    threshold: Full credibility threshold\n\nReturns:\n    Credibility factor (max 1.0)\n'''"
},
{
    "instruction": "def trending_factor(years, rate):\n    return (1 + rate) ** years",
    "response": "'''Calculate trend adjustment factor.\n\nArgs:\n    years: Time since base period\n    rate: Annual trend rate\n\nReturns:\n    Multiplicative trend factor\n'''"
},
{
    "instruction": "def unpaid_claims(ultimate, paid):\n    return ultimate - paid",
    "response": "'''Estimate unpaid claims.\n\nArgs:\n    ultimate: Projected ultimate losses\n    paid: Claims paid to date\n\nReturns:\n    Remaining unpaid amount\n'''"
},
{
    "instruction": "def has_negative_reserve(paid, reported):\n    return paid > reported",
    "response": "'''Detect negative reserve condition.\n\nArgs:\n    paid: Amount paid\n    reported: Reported incurred loss\n\nReturns:\n    True if reserve is negative\n'''"
},
{
    "instruction": "def loss_cost(per_unit_loss, exposure):\n    return per_unit_loss * exposure",
    "response": "'''Calculate expected total loss from loss cost and exposure.\n\nArgs:\n    per_unit_loss: Loss per unit (e.g., per $1000)\n    exposure: Number of exposure units\n\nReturns:\n    Expected total loss\n'''"
},
{
    "instruction": "def redact_ssns(text):\n    import re\n    return re.sub(r'\\b\\d{3}-\\d{2}-\\d{4}\\b', 'XXX-XX-XXXX', text)",
    "response": "'''Redact U.S. Social Security Numbers from text.\n\nArgs:\n    text: String that may contain SSNs\n\nReturns:\n    Text with SSNs replaced\n'''"
},
{
    "instruction": "def normalize_case_name(name):\n    return ' '.join(part.capitalize() for part in name.strip().split())",
    "response": "'''Format a legal case name in title case.\n\nArgs:\n    name: Raw case name string\n\nReturns:\n    Cleaned, title-cased case name\n'''"
},
{
    "instruction": "def extract_docket_number(text):\n    import re\n    match = re.search(r'\\b\\d{2}-[A-Z]{1,5}-\\d+\\b', text)\n    return match.group(0) if match else None",
    "response": "'''Extract a docket number from legal text.\n\nArgs:\n    text: Legal document text\n\nReturns:\n    Docket number string or None\n'''"
},
{
    "instruction": "def count_exhibits(lines):\n    return sum(1 for line in lines if line.strip().startswith('Exhibit'))",
    "response": "'''Count how many lines begin with 'Exhibit'.\n\nArgs:\n    lines: List of document lines\n\nReturns:\n    Number of exhibit references\n'''"
},
{
    "instruction": "def normalize_citation_spacing(s):\n    return ' '.join(s.replace('v.', 'v.').split())",
    "response": "'''Clean spacing and format around legal citations.\n\nArgs:\n    s: Citation string\n\nReturns:\n    Normalized citation\n'''"
},
{
    "instruction": "def get_deadline_days(filing_date, deadline_str):\n    from datetime import timedelta\n    return filing_date + timedelta(days=int(deadline_str.strip('d')))",
    "response": "'''Calculate deadline date from a filing date and offset string.\n\nArgs:\n    filing_date: Starting date\n    deadline_str: Offset string like '30d'\n\nReturns:\n    Deadline date as datetime object\n'''"
},
{
    "instruction": "def scrub_unicode_quotes(text):\n    return text.replace('“', '\"').replace('”', '\"').replace('‘', \"'\").replace('’', \"'\")",
    "response": "'''Convert curly quotes to straight ASCII quotes.\n\nArgs:\n    text: Text with typographic quotes\n\nReturns:\n    Cleaned text with straight quotes\n'''"
},
{
    "instruction": "def remove_page_headers(lines):\n    return [line for line in lines if not line.strip().startswith('Page ')]",
    "response": "'''Remove lines that appear to be page headers.\n\nArgs:\n    lines: List of document lines\n\nReturns:\n    Filtered list without headers\n'''"
},
{
    "instruction": "def has_handwritten_signature(line):\n    return 'signed by' in line.lower() and 'image' in line.lower()",
    "response": "'''Check for reference to a handwritten signature image.\n\nArgs:\n    line: Line of document text\n\nReturns:\n    True if it mentions a signature image\n'''"
},
{
    "instruction": "def detect_redline_artifacts(text):\n    return any(token in text.lower() for token in ['track changes', 'inserted', 'deleted'])",
    "response": "'''Detect common redline/editing artifacts in text.\n\nArgs:\n    text: Raw legal document text\n\nReturns:\n    True if redline indicators are found\n'''"
},
{
    "instruction": "def is_right_triangle(a: float, b: float, c: float) -> bool:\n    sides = sorted([a, b, c])\n    return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-6",
    "response": "'''Determine if three sides form a right triangle.\n\nArgs:\n    a (float): First side length\n    b (float): Second side length\n    c (float): Third side length\n\nReturns:\n    bool: True if triangle is right-angled\n'''"
},
{
    "instruction": "def triangle_area(base: float, height: float) -> float:\n    return 0.5 * base * height",
    "response": "'''Calculate the area of a triangle.\n\nArgs:\n    base (float): Length of the base\n    height (float): Height from base to apex\n\nReturns:\n    float: Area of the triangle\n'''"
},
{
    "instruction": "def polygon_internal_angle(sides: int) -> float:\n    return ((sides - 2) * 180) / sides if sides >= 3 else 0.0",
    "response": "'''Compute the internal angle of a regular polygon.\n\nArgs:\n    sides (int): Number of sides in the polygon\n\nReturns:\n    float: Internal angle in degrees\n'''"
},
{
    "instruction": "def rectangle_diagonal(width: float, height: float) -> float:\n    return (width**2 + height**2) ** 0.5",
    "response": "'''Compute the diagonal length of a rectangle.\n\nArgs:\n    width (float): Width of the rectangle\n    height (float): Height of the rectangle\n\nReturns:\n    float: Diagonal length\n'''"
},
{
    "instruction": "def circle_circumference(radius: float) -> float:\n    from math import pi\n    return 2 * pi * radius",
    "response": "'''Calculate the circumference of a circle.\n\nArgs:\n    radius (float): Radius of the circle\n\nReturns:\n    float: Circumference of the circle\n'''"
},
{
    "instruction": "def is_point_in_square(x: float, y: float, size: float) -> bool:\n    return 0 <= x <= size and 0 <= y <= size",
    "response": "'''Check if a point lies within a square starting at origin.\n\nArgs:\n    x (float): X-coordinate of the point\n    y (float): Y-coordinate of the point\n    size (float): Length of the square’s side\n\nReturns:\n    bool: True if point is inside the square\n'''"
},
{
    "instruction": "def parallelogram_area(base: float, height: float) -> float:\n    return base * height",
    "response": "'''Compute the area of a parallelogram.\n\nArgs:\n    base (float): Length of the base\n    height (float): Vertical height from base\n\nReturns:\n    float: Area of the parallelogram\n'''"
},
{
    "instruction": "def is_equilateral_triangle(a: float, b: float, c: float) -> bool:\n    return abs(a - b) < 1e-6 and abs(b - c) < 1e-6",
    "response": "'''Check if all three sides are equal.\n\nArgs:\n    a (float): First side\n    b (float): Second side\n    c (float): Third side\n\nReturns:\n    bool: True if all sides are equal\n'''"
},
{
    "instruction": "def ellipse_area(a: float, b: float) -> float:\n    from math import pi\n    return pi * a * b",
    "response": "'''Calculate the area of an ellipse.\n\nArgs:\n    a (float): Semi-major axis length\n    b (float): Semi-minor axis length\n\nReturns:\n    float: Area of the ellipse\n'''"
},
{
    "instruction": "def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:\n    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5",
    "response": "'''Compute Euclidean distance between two 2D points.\n\nArgs:\n    x1 (float): X-coordinate of first point\n    y1 (float): Y-coordinate of first point\n    x2 (float): X-coordinate of second point\n    y2 (float): Y-coordinate of second point\n\nReturns:\n    float: Distance between the points\n'''"
},
{
    "instruction": "def is_perfect_square(n: int) -> bool:\n    root = int(n ** 0.5)\n    return root * root == n",
    "response": "'''Check if an integer is a perfect square.\n\nArgs:\n    n (int): Input integer\n\nReturns:\n    bool: True if n is a perfect square\n'''"
},
{
    "instruction": "def factorial(n: int) -> int:\n    return 1 if n == 0 else n * factorial(n - 1)",
    "response": "'''Compute the factorial of a number.\n\nArgs:\n    n (int): Non-negative integer\n\nReturns:\n    int: Factorial of n\n'''"
},
{
    "instruction": "def sum_of_squares(n: int) -> int:\n    return sum(i * i for i in range(1, n + 1))",
    "response": "'''Compute the sum of squares from 1 to n.\n\nArgs:\n    n (int): Upper bound of range\n\nReturns:\n    int: Sum of squares\n'''"
},
{
    "instruction": "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "response": "'''Determine if a number is prime.\n\nArgs:\n    n (int): Number to check\n\nReturns:\n    bool: True if prime, else False\n'''"
},
{
    "instruction": "def gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)",
    "response": "'''Compute the greatest common divisor.\n\nArgs:\n    a (int): First number\n    b (int): Second number\n\nReturns:\n    int: Greatest common divisor\n'''"
},
{
    "instruction": "def lcm(a: int, b: int) -> int:\n    from math import gcd\n    return abs(a * b) // gcd(a, b) if a and b else 0",
    "response": "'''Calculate the least common multiple of two integers.\n\nArgs:\n    a (int): First integer\n    b (int): Second integer\n\nReturns:\n    int: Least common multiple\n'''"
},
{
    "instruction": "def digit_sum(n: int) -> int:\n    return sum(int(d) for d in str(abs(n)))",
    "response": "'''Sum the digits of an integer.\n\nArgs:\n    n (int): Input number\n\nReturns:\n    int: Sum of digits\n'''"
},
{
    "instruction": "def count_divisors(n: int) -> int:\n    return sum(1 for i in range(1, int(n**0.5)+1) if n % i == 0 for _ in (0, 1) if i != n // i or i * i == n)",
    "response": "'''Count the number of positive divisors of n.\n\nArgs:\n    n (int): Input number\n\nReturns:\n    int: Number of divisors\n'''"
},
{
    "instruction": "def nth_fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "response": "'''Compute the nth Fibonacci number.\n\nArgs:\n    n (int): Index in Fibonacci sequence (0-based)\n\nReturns:\n    int: The nth Fibonacci number\n'''"
},
{
    "instruction": "def is_harshad(n: int) -> bool:\n    return n % digit_sum(n) == 0 if n != 0 else False",
    "response": "'''Check if a number is a Harshad number.\n\nArgs:\n    n (int): Input number\n\nReturns:\n    bool: True if divisible by its digit sum\n'''"
},
{
    "instruction": "def normalize_title(title: str) -> str:\n    return title.lower().replace('sr.', 'senior').replace('jr.', 'junior').title()",
    "response": "'''Normalize job title casing and common abbreviations.\n\nArgs:\n    title (str): Raw job title\n\nReturns:\n    str: Cleaned, normalized title\n'''"
},
{
    "instruction": "def extract_initials(full_name: str) -> str:\n    return ''.join(word[0] for word in full_name.split() if word).upper()",
    "response": "'''Extract uppercase initials from a full name.\n\nArgs:\n    full_name (str): First + last name, etc.\n\nReturns:\n    str: Initials in uppercase\n'''"
},
{
    "instruction": "def clean_salary(s: str) -> float:\n    return float(s.replace('$', '').replace(',', '').strip())",
    "response": "'''Convert a salary string into a float.\n\nArgs:\n    s (str): String like \"$82,000\"\n\nReturns:\n    float: Parsed numeric salary\n'''"
},
{
    "instruction": "def is_contractor_email(email: str) -> bool:\n    return email.endswith('@gmail.com') or email.endswith('@outlook.com')",
    "response": "'''Check if an email likely belongs to a contractor.\n\nArgs:\n    email (str): User email address\n\nReturns:\n    bool: True if common public domain\n'''"
},
{
    "instruction": "def detect_duplicate_ids(ids: list[str]) -> bool:\n    return len(ids) != len(set(ids))",
    "response": "'''Check for duplicate user IDs in a list.\n\nArgs:\n    ids (list[str]): List of employee IDs\n\nReturns:\n    bool: True if duplicates exist\n'''"
},
{
    "instruction": "def collapse_dept_path(path: str) -> str:\n    return ' > '.join(part.strip() for part in path.split('>') if part.strip())",
    "response": "'''Clean up department hierarchy strings.\n\nArgs:\n    path (str): Department path like 'Sales >  > West Coast'\n\nReturns:\n    str: Collapsed and trimmed hierarchy\n'''"
},
{
    "instruction": "def is_valid_employee_id(emp_id: str) -> bool:\n    return emp_id.isalnum() and len(emp_id) >= 5",
    "response": "'''Validate a basic employee ID format.\n\nArgs:\n    emp_id (str): Employee identifier\n\nReturns:\n    bool: True if format is valid\n'''"
},
{
    "instruction": "def strip_honorific(name: str) -> str:\n    for prefix in ['Mr.', 'Ms.', 'Mrs.', 'Dr.']:\n        if name.startswith(prefix):\n            return name[len(prefix):].strip()\n    return name",
    "response": "'''Remove common honorifics from a name.\n\nArgs:\n    name (str): Name with possible prefix\n\nReturns:\n    str: Name without honorific\n'''"
},
{
    "instruction": "def phone_last4(phone: str) -> str:\n    digits = ''.join(filter(str.isdigit, phone))\n    return digits[-4:] if len(digits) >= 4 else ''",
    "response": "'''Extract the last 4 digits from a phone number.\n\nArgs:\n    phone (str): Raw phone number\n\nReturns:\n    str: Last 4 numeric digits\n'''"
},
{
    "instruction": "def normalize_work_location(loc: str) -> str:\n    return loc.strip().lower().replace('hq', 'headquarters').title()",
    "response": "'''Standardize formatting for work locations.\n\nArgs:\n    loc (str): Raw location string\n\nReturns:\n    str: Cleaned, title-cased location\n'''"
},
{
    "instruction": "def extract_branch_name(ref: str) -> str:\n    return ref.rsplit('/', 1)[-1]",
    "response": "'''Extract branch name from a Git ref string.\n\nArgs:\n    ref (str): Full Git ref (e.g., 'refs/heads/main')\n\nReturns:\n    str: Branch name (e.g., 'main')\n'''"
},
{
    "instruction": "def is_semver(tag: str) -> bool:\n    import re\n    return bool(re.fullmatch(r'v?\\d+\\.\\d+\\.\\d+', tag))",
    "response": "'''Check if a string matches semantic versioning.\n\nArgs:\n    tag (str): Version tag (e.g., 'v1.2.3')\n\nReturns:\n    bool: True if semver compliant\n'''"
},
{
    "instruction": "def parse_env_var(line: str) -> tuple[str, str]:\n    key, val = line.strip().split('=', 1)\n    return key.strip(), val.strip()",
    "response": "'''Split an env var assignment line into key and value.\n\nArgs:\n    line (str): Line like 'API_KEY=xyz'\n\nReturns:\n    tuple[str, str]: (key, value)\n'''"
},
{
    "instruction": "def is_docker_tag_valid(tag: str) -> bool:\n    return '/' not in tag and ':' not in tag and len(tag) <= 128",
    "response": "'''Check if a Docker tag is valid and simple.\n\nArgs:\n    tag (str): Docker image tag\n\nReturns:\n    bool: True if tag is valid\n'''"
},
{
    "instruction": "def get_commit_short_hash(sha: str) -> str:\n    return sha[:7]",
    "response": "'''Shorten a Git commit SHA to 7 characters.\n\nArgs:\n    sha (str): Full commit hash\n\nReturns:\n    str: Short hash\n'''"
},
{
    "instruction": "def canonicalize_image_name(name: str) -> str:\n    return name if ':' in name else name + ':latest'",
    "response": "'''Ensure a Docker image name includes a tag.\n\nArgs:\n    name (str): Docker image name\n\nReturns:\n    str: Name with tag, defaulting to 'latest'\n'''"
},
{
    "instruction": "def is_ci_triggered_by_pr(ref: str) -> bool:\n    return 'refs/pull/' in ref",
    "response": "'''Detect if CI run was triggered by a pull request.\n\nArgs:\n    ref (str): Git ref string\n\nReturns:\n    bool: True if pull request\n'''"
},
{
    "instruction": "def expand_matrix_var(template: str, var: str, value: str) -> str:\n    return template.replace(f'${{{var}}}', value)",
    "response": "'''Expand a GitHub Actions matrix variable in a string.\n\nArgs:\n    template (str): String with placeholder (e.g., '${{ matrix.os }}')\n    var (str): Variable name\n    value (str): Substitution value\n\nReturns:\n    str: Expanded string\n'''"
},
{
    "instruction": "def extract_artifact_id(path: str) -> str:\n    return path.split('/')[-1].split('.')[0]",
    "response": "'''Extract artifact ID from a file path.\n\nArgs:\n    path (str): Full artifact path\n\nReturns:\n    str: Artifact identifier\n'''"
},
{
    "instruction": "def sanitize_branch_name(name: str) -> str:\n    import re\n    return re.sub(r'[^a-zA-Z0-9._-]', '-', name)",
    "response": "'''Convert a Git branch name to a CI-safe string.\n\nArgs:\n    name (str): Original branch name\n\nReturns:\n    str: Sanitized name with invalid chars replaced\n'''"
},
{
    "instruction": "def is_ssh_login_line(log: str) -> bool:\n    return 'sshd' in log.lower() and 'accepted password' in log.lower()",
    "response": "'''Check if a log line indicates successful SSH login.\n\nArgs:\n    log (str): Raw log entry\n\nReturns:\n    bool: True if line shows SSH login success\n'''"
},
{
    "instruction": "def extract_ip(line: str) -> str:\n    import re\n    match = re.search(r'(\\d{1,3}\\.){3}\\d{1,3}', line)\n    return match.group(0) if match else ''",
    "response": "'''Extract an IPv4 address from a log line.\n\nArgs:\n    line (str): Log text containing IP\n\nReturns:\n    str: Extracted IP address or empty string\n'''"
},
{
    "instruction": "def is_sudo_command(log: str) -> bool:\n    return 'sudo:' in log and 'COMMAND=' in log",
    "response": "'''Detect sudo command usage in a syslog line.\n\nArgs:\n    log (str): Log message\n\nReturns:\n    bool: True if sudo command was used\n'''"
},
{
    "instruction": "def detect_curl_usage(cmd: str) -> bool:\n    return 'curl' in cmd and any(x in cmd for x in ['http://', 'https://'])",
    "response": "'''Check if a shell command uses curl to access a URL.\n\nArgs:\n    cmd (str): Shell command string\n\nReturns:\n    bool: True if curl with URL is used\n'''"
},
{
    "instruction": "def is_failed_login(log: str) -> bool:\n    return 'failed password' in log.lower() or 'authentication failure' in log.lower()",
    "response": "'''Determine if a log line indicates failed login.\n\nArgs:\n    log (str): System authentication log\n\nReturns:\n    bool: True if failed login attempt\n'''"
},
{
    "instruction": "def has_encoded_payload(s: str) -> bool:\n    return 'base64' in s or ('echo' in s and '| base64' in s)",
    "response": "'''Detect signs of encoded payloads in shell strings.\n\nArgs:\n    s (str): Shell or script content\n\nReturns:\n    bool: True if suspicious encoding is found\n'''"
},
{
    "instruction": "def extract_user_from_sudo(log: str) -> str:\n    import re\n    match = re.search(r'USER=(\\w+)', log)\n    return match.group(1) if match else ''",
    "response": "'''Extract username from a sudo audit log line.\n\nArgs:\n    log (str): Log line containing 'USER='\n\nReturns:\n    str: Extracted username or empty string\n'''"
},
{
    "instruction": "def is_reverse_shell_command(cmd: str) -> bool:\n    return 'nc' in cmd or 'bash -i' in cmd or '/dev/tcp/' in cmd",
    "response": "'''Heuristically detect a reverse shell command.\n\nArgs:\n    cmd (str): Command string from a shell log\n\nReturns:\n    bool: True if it resembles a reverse shell\n'''"
},
{
    "instruction": "def is_file_exfil_attempt(cmd: str) -> bool:\n    return any(tool in cmd for tool in ['scp', 'ftp', 'wget', 'curl']) and ('/etc/passwd' in cmd or '/var/log' in cmd)",
    "response": "'''Check for signs of sensitive file exfiltration.\n\nArgs:\n    cmd (str): Shell or command string\n\nReturns:\n    bool: True if exfiltration pattern detected\n'''"
},
{
    "instruction": "def is_suspicious_cronjob(line: str) -> bool:\n    return '@reboot' in line or '/tmp' in line or 'base64' in line",
    "response": "'''Detect suspicious patterns in a crontab line.\n\nArgs:\n    line (str): Crontab entry\n\nReturns:\n    bool: True if cron line looks suspicious\n'''"
},
{
    "instruction": "def is_valid_policy_number(p: str) -> bool:\n    import re\n    return bool(re.fullmatch(r'[A-Z]{2}-\\d{6}', p))",
    "response": "'''Validate a basic policy number format.\n\nArgs:\n    p (str): Policy number string\n\nReturns:\n    bool: True if format matches e.g., 'TX-123456'\n'''"
},
{
    "instruction": "def is_comprehensive_claim(code: str) -> bool:\n    return code.upper().startswith('COMP')",
    "response": "'''Identify if a claim code indicates comprehensive coverage.\n\nArgs:\n    code (str): Claim type code\n\nReturns:\n    bool: True if code is for comprehensive loss\n'''"
},
{
    "instruction": "def has_glass_damage(desc: str) -> bool:\n    return 'windshield' in desc.lower() or 'glass' in desc.lower()",
    "response": "'''Detect if a claim description mentions glass damage.\n\nArgs:\n    desc (str): Free-text claim description\n\nReturns:\n    bool: True if glass damage is likely\n'''"
},
{
    "instruction": "def extract_deductible_amount(s: str) -> float:\n    import re\n    match = re.search(r'\\$?(\\d{2,5})', s.replace(',', ''))\n    return float(match.group(1)) if match else 0.0",
    "response": "'''Extract a deductible amount from a messy string.\n\nArgs:\n    s (str): Raw deductible string\n\nReturns:\n    float: Parsed deductible, or 0.0 if not found\n'''"
},
{
    "instruction": "def is_fraud_flagged(notes: str) -> bool:\n    return 'siu' in notes.lower() or 'refer to fraud' in notes.lower()",
    "response": "'''Detect notes that indicate potential fraud handling.\n\nArgs:\n    notes (str): Adjuster notes or system comments\n\nReturns:\n    bool: True if fraud keywords are present\n'''"
},
{
    "instruction": "def is_total_loss(estimate: float, acv: float) -> bool:\n    return estimate >= 0.8 * acv",
    "response": "'''Determine if a vehicle is likely a total loss.\n\nArgs:\n    estimate (float): Repair estimate\n    acv (float): Actual cash value of the vehicle\n\nReturns:\n    bool: True if total loss threshold is met\n'''"
},
{
    "instruction": "def should_pay_rental(days_lost: int, rental_limit: int) -> bool:\n    return days_lost > 1 and rental_limit > 0",
    "response": "'''Check if rental reimbursement is likely warranted.\n\nArgs:\n    days_lost (int): Days vehicle is inoperable\n    rental_limit (int): Available rental coverage days\n\nReturns:\n    bool: True if rental should be paid\n'''"
},
{
    "instruction": "def parse_loss_date(date_str: str) -> str:\n    from datetime import datetime\n    return datetime.strptime(date_str, '%m/%d/%Y').date().isoformat()",
    "response": "'''Parse and reformat a loss date string.\n\nArgs:\n    date_str (str): Date in MM/DD/YYYY format\n\nReturns:\n    str: Date in ISO format (YYYY-MM-DD)\n'''"
},
{
    "instruction": "def days_since_loss(loss_date: str, today: str) -> int:\n    from datetime import datetime\n    d1 = datetime.fromisoformat(loss_date)\n    d2 = datetime.fromisoformat(today)\n    return (d2 - d1).days",
    "response": "'''Calculate number of days since the loss occurred.\n\nArgs:\n    loss_date (str): ISO-format loss date\n    today (str): ISO-format current date\n\nReturns:\n    int: Number of days since loss\n'''"
},
{
    "instruction": "def is_multi_vehicle_policy(vehicle_count: int) -> bool:\n    return vehicle_count > 1",
    "response": "'''Determine if a policy covers multiple vehicles.\n\nArgs:\n    vehicle_count (int): Number of listed vehicles\n\nReturns:\n    bool: True if policy is multi-vehicle\n'''"
},
{
    "instruction": "def normalize_vin(vin: str) -> str:\n    return vin.strip().upper().replace(' ', '')",
    "response": "'''Standardize formatting of a VIN for lookup.\n\nArgs:\n    vin (str): Raw VIN string\n\nReturns:\n    str: Cleaned, uppercase VIN with no spaces\n'''"
},
{
    "instruction": "def is_subrogation_applicable(claim_notes: str) -> bool:\n    return 'other party at fault' in claim_notes.lower() or 'pursue recovery' in claim_notes.lower()",
    "response": "'''Determine if subrogation may be relevant based on notes.\n\nArgs:\n    claim_notes (str): Adjuster or system notes\n\nReturns:\n    bool: True if indicators of subrogation appear\n'''"
},
{
    "instruction": "def match_loss_state_from_address(address: str) -> str:\n    import re\n    match = re.search(r'\\b[A-Z]{2}\\b', address.upper())\n    return match.group(0) if match else ''",
    "response": "'''Extract 2-letter state abbreviation from an address.\n\nArgs:\n    address (str): Full or partial address string\n\nReturns:\n    str: State abbreviation if found, else empty\n'''"
},
{
    "instruction": "def extract_date_range(text: str) -> tuple[str, str]:\n    import re\n    match = re.search(r'(\\d{2}/\\d{2}/\\d{4})\\s*-\\s*(\\d{2}/\\d{2}/\\d{4})', text)\n    return (match.group(1), match.group(2)) if match else ('', '')",
    "response": "'''Extract start and end dates from a date range string.\n\nArgs:\n    text (str): Text containing a range like '01/01/2023 - 01/15/2023'\n\nReturns:\n    tuple[str, str]: (start_date, end_date) or ('', '') if not found\n'''"
},
{
    "instruction": "def is_vehicle_older_than(model_year: int, threshold_year: int) -> bool:\n    return model_year < threshold_year",
    "response": "'''Determine if vehicle model year is below a threshold.\n\nArgs:\n    model_year (int): Model year of the vehicle\n    threshold_year (int): Year to compare against\n\nReturns:\n    bool: True if vehicle is older than threshold\n'''"
},
{
    "instruction": "def estimate_medical_payout(visit_count: int, avg_cost: float, policy_limit: float) -> float:\n    est = visit_count * avg_cost\n    return min(est, policy_limit)",
    "response": "'''Estimate total medical payout up to policy limit.\n\nArgs:\n    visit_count (int): Number of medical visits\n    avg_cost (float): Average cost per visit\n    policy_limit (float): Medical coverage limit\n\nReturns:\n    float: Estimated covered payout\n'''"
},
{
    "instruction": "def likely_total_loss(estimate: float, acv: float, threshold: float = 0.75) -> bool:\n    return acv > 0 and estimate / acv >= threshold",
    "response": "'''Check if repair estimate suggests a total loss.\n\nArgs:\n    estimate (float): Repair cost\n    acv (float): Actual cash value of vehicle\n    threshold (float): Ratio to trigger total loss (default 0.75)\n\nReturns:\n    bool: True if above threshold\n'''"
},
{
    "instruction": "def is_policy_active(as_of: str, effective: str, expiration: str) -> bool:\n    from datetime import datetime\n    d = datetime.fromisoformat(as_of)\n    return datetime.fromisoformat(effective) <= d <= datetime.fromisoformat(expiration)",
    "response": "'''Check if a policy is active on a specific date.\n\nArgs:\n    as_of (str): Date to evaluate (ISO format)\n    effective (str): Policy start date\n    expiration (str): Policy end date\n\nReturns:\n    bool: True if policy is active on that date\n'''"
},
{
    "instruction": "def calculate_recovery_ratio(recovered: float, paid: float) -> float:\n    return round(recovered / paid, 2) if paid else 0.0",
    "response": "'''Compute recovery-to-paid ratio for subrogation.\n\nArgs:\n    recovered (float): Amount recovered from third party\n    paid (float): Total paid on claim\n\nReturns:\n    float: Recovery ratio rounded to 2 decimals\n'''"
},
{
    "instruction": "def needs_property_inspection(claim_type: str, policy_age: int) -> bool:\n    return claim_type.lower() == 'property' and policy_age >= 5",
    "response": "'''Determine if property claim may warrant inspection.\n\nArgs:\n    claim_type (str): Type of claim (e.g., 'auto', 'property')\n    policy_age (int): Years since policy started\n\nReturns:\n    bool: True if inspection should be considered\n'''"
},
{
    "instruction": (
        "def execute_pipeline(pipeline, *args, **kwargs):\n"
        "    result = args\n"
        "    for step in pipeline:\n"
        "        result = (step(*result, **kwargs),)\n"
        "    return result[0]"
    ),
    "response": (
        '"""Run a sequence of functions with shared args/kwargs.\n\n'
        'Args:\n'
        '    pipeline: List of callables to run in order\n'
        '    *args: Positional arguments for the first step\n'
        '    **kwargs: Keyword arguments passed to all steps\n\n'
        'Returns:\n'
        '    Final output after running through all steps\n'
        '"""'
    )
},
{
    "instruction": (
        "def timed(func, *args, **kwargs):\n"
        "    import time\n"
        "    start = time.perf_counter()\n"
        "    result = func(*args, **kwargs)\n"
        "    end = time.perf_counter()\n"
        "    print(f\"Execution took {end - start:.4f} seconds\")\n"
        "    return result"
    ),
    "response": (
        '"""Execute a function and log its runtime.\n\n'
        'Args:\n'
        '    func: Callable to execute\n'
        '    *args: Positional arguments for the function\n'
        '    **kwargs: Keyword arguments for the function\n\n'
        'Returns:\n'
        '    Result of the function call\n'
        '"""'
    )
},
{
    "instruction": (
        "def call_with_retries(func, max_attempts=3, *args, **kwargs):\n"
        "    for attempt in range(max_attempts):\n"
        "        try:\n"
        "            return func(*args, **kwargs)\n"
        "        except Exception as e:\n"
        "            if attempt == max_attempts - 1:\n"
        "                raise e"
    ),
    "response": (
        '"""Retry a function multiple times on exception.\n\n'
        'Args:\n'
        '    func: Function to retry\n'
        '    max_attempts: Max retry attempts\n'
        '    *args: Positional args for the function\n'
        '    **kwargs: Keyword args for the function\n\n'
        'Returns:\n'
        '    Result of the function call\n\n'
        'Raises:\n'
        '    Exception: If all attempts fail\n'
        '"""'
    )
},
{
    "instruction": (
        "def sanitize_and_call(func, *args, **kwargs):\n"
        "    args = [a.strip() if isinstance(a, str) else a for a in args]\n"
        "    kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}\n"
        "    return func(*args, **kwargs)"
    ),
    "response": (
        '"""Trim string inputs before calling a function.\n\n'
        'Args:\n'
        '    func: Callable to invoke\n'
        '    *args: Positional arguments (strings trimmed)\n'
        '    **kwargs: Keyword arguments (strings trimmed)\n\n'
        'Returns:\n'
        '    Result of the function call\n'
        '"""'
    )
},
{
    "instruction": (
        "def warn_unused_kwargs(func_name, allowed_keys, **kwargs):\n"
        "    for key in kwargs:\n"
        "        if key not in allowed_keys:\n"
        "            print(f\"Warning: {key} not used in {func_name}()\")"
    ),
    "response": (
        '"""Warn about unexpected keyword arguments.\n\n'
        'Args:\n'
        '    func_name: Name of the calling function\n'
        '    allowed_keys: List of allowed keyword argument names\n'
        '    **kwargs: All keyword arguments received\n\n'
        'Returns:\n'
        '    None\n'
        '"""'
    )
},
{
    "instruction": (
        "def normalize_scores(scores):\n"
        "    if not scores:\n"
        "        return []\n"
        "    max_score = max(scores)\n"
        "    if max_score == 0:\n"
        "        return [0] * len(scores)\n"
        "    return [s / max_score for s in scores]"
    ),
    "response": (
        '"""Normalize a list of numeric scores to range [0, 1].\n\n'
        'Args:\n'
        '    scores: List of numeric values\n\n'
        'Returns:\n'
        '    List of normalized values, or empty list if input is empty\n'
        '"""'
    )
},
{
    "instruction": (
        "def read_config_file(path):\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        lines = f.readlines()\n"
        "    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]"
    ),
    "response": (
        '"""Read a config file, ignoring blank lines and comments.\n\n'
        'Args:\n'
        '    path: File path to config file\n\n'
        'Returns:\n'
        '    List of non-comment, non-empty lines\n'
        '"""'
    )
},
{
    "instruction": (
        "def find_largest_file(paths):\n"
        "    import os\n"
        "    largest = None\n"
        "    max_size = -1\n"
        "    for path in paths:\n"
        "        size = os.path.getsize(path)\n"
        "        if size > max_size:\n"
        "            max_size = size\n"
        "            largest = path\n"
        "    return largest"
    ),
    "response": (
        '"""Return the path to the largest file by size.\n\n'
        'Args:\n'
        '    paths: List of file paths\n\n'
        'Returns:\n'
        '    Path of the largest file, or None if list is empty\n'
        '"""'
    )
},
{
    "instruction": (
        "def split_on_delimiters(text, delimiters):\n"
        "    import re\n"
        "    pattern = '|'.join(map(re.escape, delimiters))\n"
        "    return [part.strip() for part in re.split(pattern, text) if part.strip()]"
    ),
    "response": (
        '"""Split a string on multiple delimiters.\n\n'
        'Args:\n'
        '    text: Input string\n'
        '    delimiters: List of string delimiters\n\n'
        'Returns:\n'
        '    List of cleaned substrings\n'
        '"""'
    )
},
{
    "instruction": (
        "def convert_units(value, from_unit, to_unit):\n"
        "    conversions = {\n"
        "        ('km', 'm'): 1000,\n"
        "        ('m', 'km'): 1 / 1000,\n"
        "        ('cm', 'm'): 1 / 100,\n"
        "        ('m', 'cm'): 100,\n"
        "    }\n"
        "    factor = conversions.get((from_unit, to_unit))\n"
        "    if factor is None:\n"
        "        raise ValueError('Unsupported conversion')\n"
        "    return value * factor"
    ),
    "response": (
        '"""Convert value between supported metric units.\n\n'
        'Args:\n'
        '    value: Numeric value to convert\n'
        '    from_unit: Source unit (e.g., "km")\n'
        '    to_unit: Target unit (e.g., "m")\n\n'
        'Returns:\n'
        '    Converted value\n\n'
        'Raises:\n'
        '    ValueError: If the conversion pair is unsupported\n'
        '"""'
    )
},
{
    "instruction": (
        "def impute_missing(df, strategy='mean'):\n"
        "    import pandas as pd\n"
        "    result = df.copy()\n"
        "    for col in result.select_dtypes(include='number').columns:\n"
        "        if strategy == 'mean':\n"
        "            fill_value = result[col].mean()\n"
        "        elif strategy == 'median':\n"
        "            fill_value = result[col].median()\n"
        "        else:\n"
        "            raise ValueError('Unsupported strategy')\n"
        "        result[col].fillna(fill_value, inplace=True)\n"
        "    return result"
    ),
    "response": (
        '"""Impute missing numeric values in a DataFrame.\n\n'
        'Args:\n'
        '    df: Input DataFrame\n'
        '    strategy: Imputation strategy ("mean" or "median")\n\n'
        'Returns:\n'
        '    DataFrame with missing values filled\n\n'
        'Raises:\n'
        '    ValueError: If the strategy is unsupported\n'
        '"""'
    )
},
{
    "instruction": (
        "def clip_outliers(df, columns, lower=0.01, upper=0.99):\n"
        "    result = df.copy()\n"
        "    for col in columns:\n"
        "        q_low = result[col].quantile(lower)\n"
        "        q_high = result[col].quantile(upper)\n"
        "        result[col] = result[col].clip(q_low, q_high)\n"
        "    return result"
    ),
    "response": (
        '"""Clip outliers in numeric columns using quantiles.\n\n'
        'Args:\n'
        '    df: DataFrame containing data\n'
        '    columns: List of numeric column names\n'
        '    lower: Lower quantile threshold\n'
        '    upper: Upper quantile threshold\n\n'
        'Returns:\n'
        '    DataFrame with values clipped to quantile range\n'
        '"""'
    )
},
{
    "instruction": (
        "def categorize_age(age):\n"
        "    if age < 18:\n"
        "        return 'child'\n"
        "    elif age < 65:\n"
        "        return 'adult'\n"
        "    else:\n"
        "        return 'senior'"
    ),
    "response": (
        '"""Categorize age into demographic bins.\n\n'
        'Args:\n'
        '    age: Integer or float age value\n\n'
        'Returns:\n'
        '    String category: "child", "adult", or "senior"\n'
        '"""'
    )
},
{
    "instruction": (
        "def scale_features(df, columns):\n"
        "    from sklearn.preprocessing import StandardScaler\n"
        "    scaler = StandardScaler()\n"
        "    df[columns] = scaler.fit_transform(df[columns])\n"
        "    return df"
    ),
    "response": (
        '"""Standardize selected columns in a DataFrame.\n\n'
        'Args:\n'
        '    df: Input DataFrame\n'
        '    columns: List of column names to scale\n\n'
        'Returns:\n'
        '    DataFrame with standardized columns\n'
        '"""'
    )
},
{
    "instruction": (
        "def get_class_distribution(y):\n"
        "    import collections\n"
        "    return dict(collections.Counter(y))"
    ),
    "response": (
        '"""Return class label distribution from a sequence.\n\n'
        'Args:\n'
        '    y: Iterable of class labels\n\n'
        'Returns:\n'
        '    Dictionary mapping label to count\n'
        '"""'
    )
},
{
    "instruction": (
        "def drop_high_missing(df, threshold=0.5):\n"
        "    missing_ratio = df.isna().mean()\n"
        "    to_drop = missing_ratio[missing_ratio > threshold].index\n"
        "    return df.drop(columns=to_drop)"
    ),
    "response": (
        '"""Drop columns with missing value ratio above a threshold.\n\n'
        'Args:\n'
        '    df: Input DataFrame\n'
        '    threshold: Fraction above which columns are dropped\n\n'
        'Returns:\n'
        '    DataFrame with selected columns removed\n'
        '"""'
    )
},
{
    "instruction": (
        "def encode_binary(df, column):\n"
        "    mapping = {df[column].unique()[0]: 0, df[column].unique()[1]: 1}\n"
        "    df[column] = df[column].map(mapping)\n"
        "    return df"
    ),
    "response": (
        '"""Convert a binary categorical column to 0/1 encoding.\n\n'
        'Args:\n'
        '    df: Input DataFrame\n'
        '    column: Name of the binary column\n\n'
        'Returns:\n'
        '    DataFrame with encoded column\n\n'
        'Raises:\n'
        '    IndexError: If the column has fewer than two unique values\n'
        '"""'
    )
},
{
"instruction": (
    "def get_top_categories(series, top_n=5):\n"
    "    counts = series.value_counts().nlargest(top_n)\n"
    "    return counts.index.tolist()"
),
"response": (
    '"""Return the most frequent categorical values.\n\n'
    'Args:\n'
    '    series: Pandas Series of categorical data\n'
    '    top_n: Number of top categories to return\n\n'
    'Returns:\n'
    '    List of most common categories\n'
    '"""'
)
},
{
"instruction": (
    "def remove_constant_columns(df):\n"
    "    return df.loc[:, df.nunique(dropna=False) > 1]"
),
"response": (
    '"""Drop columns with only a single unique value.\n\n'
    'Args:\n'
    '    df: Input DataFrame\n\n'
    'Returns:\n'
    '    DataFrame without constant-valued columns\n'
    '"""'
)
},
{
"instruction": (
    "def label_encode_series(series):\n"
    "    from sklearn.preprocessing import LabelEncoder\n"
    "    encoder = LabelEncoder()\n"
    "    return encoder.fit_transform(series)"
),
"response": (
    '"""Apply label encoding to a categorical Series.\n\n'
    'Args:\n'
    '    series: Pandas Series of string or categorical values\n\n'
    'Returns:\n'
    '    Numpy array of encoded integers\n'
    '"""'
)
},
{
"instruction": (
    "def summarize_numeric_columns(df):\n"
    "    desc = df.select_dtypes(include='number').describe().T\n"
    "    desc['missing'] = df.isna().sum()\n"
    "    return desc"
),
"response": (
    '"""Generate descriptive statistics for numeric columns.\n\n'
    'Args:\n'
    '    df: Input DataFrame\n\n'
    'Returns:\n'
    '    DataFrame summary with count, mean, std, min/max, and missing\n'
    '"""'
)
},
{
"instruction": (
    "def check_multicollinearity(df, threshold=0.9):\n"
    "    corr = df.corr().abs()\n"
    "    upper = corr.where(~corr.columns.duplicated(), 0).where(~(corr < threshold))\n"
    "    pairs = [(c1, c2) for c1 in upper.columns for c2 in upper.index if c1 != c2 and upper.loc[c2, c1] > threshold]\n"
    "    return list(set(tuple(sorted(p)) for p in pairs))"
),
"response": (
    '"""Detect highly correlated column pairs.\n\n'
    'Args:\n'
    '    df: DataFrame of numeric features\n'
    '    threshold: Absolute correlation threshold for flagging\n\n'
    'Returns:\n'
    '    List of (col1, col2) tuples with correlation > threshold\n'
    '"""'
)
},
{
"instruction": (
    "def get_column_type_summary(df):\n"
    "    return df.dtypes.value_counts().to_dict()"
),
"response": (
    '"""Count occurrences of each column dtype in a DataFrame.\n\n'
    'Args:\n'
    '    df: Input DataFrame\n\n'
    'Returns:\n'
    '    Dictionary mapping dtypes to count\n'
    '"""'
)
},
{
"instruction": (
    "def calculate_gini_impurity(values):\n"
    "    from collections import Counter\n"
    "    total = len(values)\n"
    "    if total == 0:\n"
    "        return 0.0\n"
    "    counts = Counter(values)\n"
    "    probs = [v / total for v in counts.values()]\n"
    "    return 1 - sum(p**2 for p in probs)"
),
"response": (
    '"""Compute Gini impurity for a list of class labels.\n\n'
    'Args:\n'
    '    values: List or Series of categorical class labels\n\n'
    'Returns:\n'
    '    Gini impurity score (float between 0 and 1)\n'
    '"""'
)
},

{
    "instruction": (
        "def extract_hashtags(text: str) -> list[str]:\n"
        "    import re\n"
        "    return re.findall(r'#\\w+', text)"
    ),
    "response": (
        '"""Extract hashtags from a text string.\n\n'
        'Args:\n'
        '    text (str): The input text containing hashtags\n\n'
        'Returns:\n'
        '    list[str]: List of hashtags found in the text\n"""'
    )
},
{
    "instruction": (
        "def filter_posts_by_user(posts: list[dict], username: str) -> list[dict]:\n"
        "    return [post for post in posts if post.get('user') == username]"
    ),
    "response": (
        '"""Filter posts to include only those from a specific user.\n\n'
        'Args:\n'
        '    posts (list[dict]): List of post objects as dictionaries\n'
        '    username (str): Username to filter by\n\n'
        'Returns:\n'
        '    list[dict]: Filtered list of posts authored by the specified user\n"""'
    )
},
{
    "instruction": (
        "def get_average_likes(posts: list[dict]) -> float:\n"
        "    likes = [post.get('likes', 0) for post in posts]\n"
        "    return sum(likes) / len(likes) if likes else 0.0"
    ),
    "response": (
        '"""Compute the average number of likes from a list of posts.\n\n'
        'Args:\n'
        '    posts (list[dict]): List of post dictionaries with "likes" keys\n\n'
        'Returns:\n'
        '    float: Average number of likes, or 0.0 if list is empty\n"""'
    )
},
{
    "instruction": (
        "def parse_mentions(text: str) -> list[str]:\n"
        "    import re\n"
        "    return re.findall(r'@\\w+', text)"
    ),
    "response": (
        '"""Extract user mentions from a string.\n\n'
        'Args:\n'
        '    text (str): Input text with potential @mentions\n\n'
        'Returns:\n'
        '    list[str]: List of mention strings starting with @\n"""'
    )
},
{
    "instruction": (
        "def summarize_engagement(posts: list[dict]) -> dict[str, float]:\n"
        "    total = len(posts)\n"
        "    likes = sum(p.get('likes', 0) for p in posts)\n"
        "    shares = sum(p.get('shares', 0) for p in posts)\n"
        "    comments = sum(p.get('comments', 0) for p in posts)\n"
        "    return {\n"
        "        'avg_likes': likes / total if total else 0.0,\n"
        "        'avg_shares': shares / total if total else 0.0,\n"
        "        'avg_comments': comments / total if total else 0.0\n"
        "    }"
    ),
    "response": (
        '"""Summarize average engagement metrics for a list of posts.\n\n'
        'Args:\n'
        '    posts (list[dict]): List of post dictionaries with "likes", "shares", and "comments"\n\n'
        'Returns:\n'
        '    dict[str, float]: Dictionary with average likes, shares, and comments\n"""'
    )
},

{
    "instruction": (
        "def calculate_sales_tax(amount: float, rate: float) -> float:\n"
        "    return round(amount * rate, 2)"
    ),
    "response": (
        '"""Calculate sales tax for a given amount.\n\n'
        'Args:\n'
        '    amount (float): Pre-tax amount\n'
        '    rate (float): Tax rate as a decimal (e.g., 0.075 for 7.5%)\n\n'
        'Returns:\n'
        '    float: Calculated sales tax rounded to 2 decimal places\n"""'
    )
},
{
    "instruction": (
        "def apply_standard_deduction(income: float, deduction: float = 13850.0) -> float:\n"
        "    return max(0.0, income - deduction)"
    ),
    "response": (
        '"""Apply the standard deduction to income.\n\n'
        'Args:\n'
        '    income (float): Gross income\n'
        '    deduction (float): Standard deduction amount (default is $13,850)\n\n'
        'Returns:\n'
        '    float: Taxable income after deduction\n"""'
    )
},
{
    "instruction": (
        "def filter_deductible_expenses(expenses: list[dict]) -> list[dict]:\n"
        "    return [e for e in expenses if e.get('deductible') is True]"
    ),
    "response": (
        '"""Return only deductible expenses from a list.\n\n'
        'Args:\n'
        '    expenses (list[dict]): List of expense records with a "deductible" field\n\n'
        'Returns:\n'
        '    list[dict]: Expenses where deductible is explicitly True\n"""'
    )
},
{
    "instruction": (
        "def compute_invoice_total(items: list[dict]) -> float:\n"
        "    return sum(item['quantity'] * item['unit_price'] for item in items)"
    ),
    "response": (
        '"""Compute the total for an invoice based on line items.\n\n'
        'Args:\n'
        '    items (list[dict]): Each item must have "quantity" and "unit_price"\n\n'
        'Returns:\n'
        '    float: Total invoice amount\n"""'
    )
},
{
    "instruction": (
        "def calculate_amortization(principal: float, annual_rate: float, years: int) -> float:\n"
        "    r = annual_rate / 12\n"
        "    n = years * 12\n"
        "    return round(principal * r / (1 - (1 + r) ** -n), 2)"
    ),
    "response": (
        '"""Calculate monthly payment for a fixed-rate loan.\n\n'
        'Args:\n'
        '    principal (float): Initial loan amount\n'
        '    annual_rate (float): Annual interest rate (e.g., 0.05 for 5%)\n'
        '    years (int): Loan term in years\n\n'
        'Returns:\n'
        '    float: Fixed monthly payment amount rounded to 2 decimals\n"""'
    )
},
{
    "instruction": (
        "def summarize_by_category(transactions: list[dict]) -> dict[str, float]:\n"
        "    from collections import defaultdict\n"
        "    summary = defaultdict(float)\n"
        "    for tx in transactions:\n"
        "        summary[tx['category']] += tx['amount']\n"
        "    return dict(summary)"
    ),
    "response": (
        '"""Aggregate transaction amounts by category.\n\n'
        'Args:\n'
        '    transactions (list[dict]): Each transaction must have "category" and "amount"\n\n'
        'Returns:\n'
        '    dict[str, float]: Mapping of category to total amount spent\n"""'
    )
},
{
    "instruction": (
        "def calculate_bmi(weight_kg: float, height_cm: float) -> float:\n"
        "    height_m = height_cm / 100\n"
        "    return round(weight_kg / (height_m ** 2), 1)"
    ),
    "response": (
        '"""Calculate Body Mass Index (BMI).\n\n'
        'Args:\n'
        '    weight_kg (float): Weight in kilograms\n'
        '    height_cm (float): Height in centimeters\n\n'
        'Returns:\n'
        '    float: BMI rounded to 1 decimal place\n"""'
    )
},
{
    "instruction": (
        "def estimate_daily_water_intake(weight_kg: float, activity_minutes: int = 30) -> float:\n"
        "    base = weight_kg * 0.03\n"
        "    activity = activity_minutes * 0.012\n"
        "    return round(base + activity, 2)"
    ),
    "response": (
        '"""Estimate daily water intake in liters.\n\n'
        'Args:\n'
        '    weight_kg (float): Body weight in kilograms\n'
        '    activity_minutes (int): Minutes of physical activity per day (default is 30)\n\n'
        'Returns:\n'
        '    float: Recommended daily water intake in liters\n"""'
    )
},
{
    "instruction": (
        "def calculate_macros(calories: int, protein_pct: float, fat_pct: float) -> dict[str, float]:\n"
        "    protein_g = (calories * protein_pct) / 4\n"
        "    fat_g = (calories * fat_pct) / 9\n"
        "    carbs_g = (calories - (protein_g * 4 + fat_g * 9)) / 4\n"
        "    return {'protein_g': round(protein_g), 'fat_g': round(fat_g), 'carbs_g': round(carbs_g)}"
    ),
    "response": (
        '"""Break down calories into macronutrients.\n\n'
        'Args:\n'
        '    calories (int): Total daily calorie goal\n'
        '    protein_pct (float): Fraction of calories from protein (e.g., 0.3)\n'
        '    fat_pct (float): Fraction of calories from fat (e.g., 0.25)\n\n'
        'Returns:\n'
        '    dict[str, float]: Grams of protein, fat, and carbs\n"""'
    )
},
{
    "instruction": (
        "def heart_rate_zone(age: int, resting_hr: int) -> dict[str, tuple[int, int]]:\n"
        "    max_hr = 220 - age\n"
        "    reserve = max_hr - resting_hr\n"
        "    zones = {\n"
        "        'zone_1': (round(resting_hr + 0.5 * reserve), round(resting_hr + 0.6 * reserve)),\n"
        "        'zone_2': (round(resting_hr + 0.6 * reserve), round(resting_hr + 0.7 * reserve)),\n"
        "        'zone_3': (round(resting_hr + 0.7 * reserve), round(resting_hr + 0.8 * reserve)),\n"
        "        'zone_4': (round(resting_hr + 0.8 * reserve), round(resting_hr + 0.9 * reserve)),\n"
        "        'zone_5': (round(resting_hr + 0.9 * reserve), round(max_hr))\n"
        "    }\n"
        "    return zones"
    ),
    "response": (
        '"""Calculate heart rate training zones using Karvonen formula.\n\n'
        'Args:\n'
        '    age (int): Age in years\n'
        '    resting_hr (int): Resting heart rate in bpm\n\n'
        'Returns:\n'
        '    dict[str, tuple[int, int]]: Target HR zones as bpm ranges\n"""'
    )
},
{
    "instruction": (
        "def is_caloric_deficit(intake: float, expenditure: float, threshold: float = 200.0) -> bool:\n"
        "    return (expenditure - intake) >= threshold"
    ),
    "response": (
        '"""Check if intake is sufficiently below expenditure for weight loss.\n\n'
        'Args:\n'
        '    intake (float): Daily calorie intake\n'
        '    expenditure (float): Daily calorie burn estimate\n'
        '    threshold (float): Minimum deficit to count as weight-loss-promoting\n\n'
        'Returns:\n'
        '    bool: True if caloric deficit exceeds threshold\n"""'
    )
},

{
    "instruction": (
        "def calculate_ore_grade(weight_mineral: float, weight_sample: float) -> float:\n"
        "    return round((weight_mineral / weight_sample) * 100, 2)"
    ),
    "response": (
        '"""Calculate ore grade as a percentage.\n\n'
        'Args:\n'
        '    weight_mineral (float): Weight of the target mineral in grams\n'
        '    weight_sample (float): Total weight of the sample in grams\n\n'
        'Returns:\n'
        '    float: Ore grade percentage, rounded to two decimals\n"""'
    )
},
{
    "instruction": (
        "def estimate_haulage_volume(truck_capacity_t: float, trips_per_day: int, days: int) -> float:\n"
        "    return round(truck_capacity_t * trips_per_day * days, 2)"
    ),
    "response": (
        '"""Estimate total haulage volume over a period.\n\n'
        'Args:\n'
        '    truck_capacity_t (float): Capacity per truck in tonnes\n'
        '    trips_per_day (int): Number of trips made daily\n'
        '    days (int): Total number of operating days\n\n'
        'Returns:\n'
        '    float: Estimated tonnage hauled\n"""'
    )
},
{
    "instruction": (
        "def is_within_dust_limit(reading: float, threshold: float = 50.0) -> bool:\n"
        "    return reading <= threshold"
    ),
    "response": (
        '"""Check if dust particle reading is within regulatory limit.\n\n'
        'Args:\n'
        '    reading (float): Measured dust concentration in µg/m³\n'
        '    threshold (float): Maximum allowable concentration (default 50.0)\n\n'
        'Returns:\n'
        '    bool: True if within limit, False if over\n"""'
    )
},
{
    "instruction": (
        "def average_downtime(events: list[dict[str, float]]) -> float:\n"
        "    durations = [e['duration_hrs'] for e in events if 'duration_hrs' in e]\n"
        "    return round(sum(durations) / len(durations), 2) if durations else 0.0"
    ),
    "response": (
        '"""Compute average equipment downtime from event records.\n\n'
        'Args:\n'
        '    events (list[dict[str, float]]): List of events with "duration_hrs" field\n\n'
        'Returns:\n'
        '    float: Average downtime in hours, or 0.0 if none\n"""'
    )
},
{
    "instruction": (
        "def flag_low_recovery(yield_pct: float, threshold: float = 85.0) -> bool:\n"
        "    return yield_pct < threshold"
    ),
    "response": (
        '"""Flag if metallurgical recovery is below target.\n\n'
        'Args:\n'
        '    yield_pct (float): Percent of mineral recovered from ore\n'
        '    threshold (float): Minimum acceptable recovery percentage\n\n'
        'Returns:\n'
        '    bool: True if below threshold, else False\n"""'
    )
},
{
    "instruction": (
        "def calculate_strip_ratio(overburden_volume: float, ore_volume: float) -> float:\n"
        "    if ore_volume == 0:\n"
        "        raise ValueError('Ore volume must be greater than zero')\n"
        "    return round(overburden_volume / ore_volume, 2)"
    ),
    "response": (
        '"""Calculate the strip ratio for a mine.\n\n'
        'Args:\n'
        '    overburden_volume (float): Volume of waste material removed\n'
        '    ore_volume (float): Volume of ore extracted\n\n'
        'Returns:\n'
        '    float: Strip ratio (waste : ore), rounded to 2 decimals\n\n'
        'Raises:\n'
        '    ValueError: If ore volume is zero\n"""'
    )
},
{
    "instruction": (
        "def word_count(text):\n"
        "    return len(text.split())"
    ),
    "response": (
        '"""Count the number of words in a text.\n\n'
        'Args:\n'
        '    text: Input article or paragraph\n\n'
        'Returns:\n'
        '    Total number of words\n"""'
    )
},
{
    "instruction": (
        "def is_featured(article: dict) -> bool:\n"
        "    return article.get('category') == 'Feature' and article.get('priority') >= 8"
    ),
    "response": (
        '"""Determine if an article should be marked as featured.\n\n'
        'Args:\n'
        '    article (dict): Dictionary with article metadata\n\n'
        'Returns:\n'
        '    bool: True if featured, False otherwise\n"""'
    )
},
{
    "instruction": (
        "def estimate_read_time(text: str, wpm: int = 220) -> int:\n"
        "    words = len(text.split())\n"
        "    return max(1, round(words / wpm))"
    ),
    "response": (
        '"""Estimate read time based on word count.\n\n'
        'Args:\n'
        '    text (str): Article body text\n'
        '    wpm (int): Average reading speed (default is 220)\n\n'
        'Returns:\n'
        '    int: Estimated reading time in minutes\n"""'
    )
},
{
    "instruction": (
        "def get_authors_by_section(articles, section):\n"
        "    return list({a['author'] for a in articles if a.get('section') == section})"
    ),
    "response": (
        '"""List unique authors writing for a specific section.\n\n'
        'Args:\n'
        '    articles: List of article dictionaries\n'
        '    section: Target section name (e.g., "Sports")\n\n'
        'Returns:\n'
        '    List of unique author names\n"""'
    )
},
{
    "instruction": (
        "def paywall_required(user: dict, article: dict) -> bool:\n"
        "    if user.get('is_subscriber'):\n"
        "        return False\n"
        "    return article.get('access') == 'subscriber-only'"
    ),
    "response": (
        '"""Determine if a paywall should be enforced.\n\n'
        'Args:\n'
        '    user (dict): Dictionary with user status\n'
        '    article (dict): Dictionary with article access metadata\n\n'
        'Returns:\n'
        '    bool: True if paywall is needed, False otherwise\n"""'
    )
},
{
    "instruction": (
        "def schedule_publication_time(day: str, hour: int) -> str:\n"
        "    from datetime import datetime, timedelta\n"
        "    weekday_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n"
        "    today = datetime.now()\n"
        "    target_weekday = weekday_map.index(day)\n"
        "    days_ahead = (target_weekday - today.weekday()) % 7\n"
        "    publish_date = today + timedelta(days=days_ahead)\n"
        "    return publish_date.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()"
    ),
    "response": (
        '"""Calculate the next datetime for article publication.\n\n'
        'Args:\n'
        '    day (str): Day of week for publication (e.g., "Friday")\n'
        '    hour (int): Hour of day in 24h format (e.g., 9 for 9:00 AM)\n\n'
        'Returns:\n'
        '    str: ISO-formatted datetime string of next publication\n"""'
    )
},

{
    "instruction": (
        "def validate_password(password: str) -> bool:\n"
        "    import re\n"
        "    if len(password) < 8:\n"
        "        return False\n"
        "    if not re.search(r'[A-Z]', password):\n"
        "        return False\n"
        "    if not re.search(r'[a-z]', password):\n"
        "        return False\n"
        "    if not re.search(r'[0-9]', password):\n"
        "        return False\n"
        "    return True"
    ),
    "response": (
        '"""Check password strength using basic rules.\n\n'
        'Args:\n'
        '    password (str): Raw password string\n\n'
        'Returns:\n'
        '    bool: True if password meets strength criteria\n"""'
    )
},
{
    "instruction": (
        "def build_pagination_links(base_url: str, page: int, total_pages: int) -> dict[str, str]:\n"
        "    links = {}\n"
        "    if page > 1:\n"
        "        links['prev'] = f\"{base_url}?page={page - 1}\"\n"
        "    if page < total_pages:\n"
        "        links['next'] = f\"{base_url}?page={page + 1}\"\n"
        "    return links"
    ),
    "response": (
        '"""Generate previous/next pagination URLs.\n\n'
        'Args:\n'
        '    base_url (str): Base endpoint URL\n'
        '    page (int): Current page number\n'
        '    total_pages (int): Total number of available pages\n\n'
        'Returns:\n'
        '    dict[str, str]: Dictionary with "prev" and/or "next" URLs\n"""'
    )
},
{
    "instruction": (
        "def is_valid_email(email: str) -> bool:\n"
        "    import re\n"
        "    return bool(re.match(r'^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$', email))"
    ),
    "response": (
        '"""Validate that an email has a proper format.\n\n'
        'Args:\n'
        '    email (str): Email address to validate\n\n'
        'Returns:\n'
        '    bool: True if email format is valid\n"""'
    )
},
{
    "instruction": (
        "def extract_token(headers: dict[str, str]) -> str | None:\n"
        "    auth = headers.get('Authorization', '')\n"
        "    if auth.startswith('Bearer '):\n"
        "        return auth[7:]\n"
        "    return None"
    ),
    "response": (
        '"""Extract bearer token from HTTP headers.\n\n'
        'Args:\n'
        '    headers (dict[str, str]): Request headers\n\n'
        'Returns:\n'
        '    str | None: Token string if present, else None\n"""'
    )
},
{
    "instruction": (
        "def redirect_if_unauthorized(user_id: int | None, login_url: str) -> str:\n"
        "    if user_id is None:\n"
        "        return f\"302 Redirect: {login_url}\"\n"
        "    return \"200 OK\""
    ),
    "response": (
        '"""Return redirect string if user is not authenticated.\n\n'
        'Args:\n'
        '    user_id (int | None): ID of logged-in user, or None if not authenticated\n'
        '    login_url (str): URL to redirect to if unauthenticated\n\n'
        'Returns:\n'
        '    str: Redirect or success message\n"""'
    )
}
]