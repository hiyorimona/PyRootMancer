import unittest
from unittest.mock import patch
from prc.utils.configuration import get_project_root, base_folder


class TestGetProjectRoot(unittest.TestCase):

    @patch('os.path.join', return_value='2023-24d-fai2-adsai-group-cv5')
    def test_get_project_root(self, mock_join):
        expected_result = get_project_root()
        print(base_folder)
        repository_name = "2023-24d-fai2-adsai-group-cv5"
        self.assertEqual(get_project_root(repository_name), expected_result)


if __name__ == '__main__':
    unittest.main()
