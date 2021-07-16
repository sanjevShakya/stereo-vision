import cv2 as cv
import numpy as np

class CVFileUtil:
    output_file = ''
    file_storage = None

    def __init__(self, output_file):
        self.output_file = output_file
        return None

    def open_read_mode(self):
        self.file_storage = cv.FileStorage(
            self.output_file, cv.FILE_STORAGE_READ)

    def open_write_mode(self):

        self.file_storage = cv.FileStorage(
            self.output_file, cv.FILE_STORAGE_WRITE)

    def write_matrix(self, key, row):
        self.file_storage.write(key, np.array(row))

    def read_matrix(self, key):
        return self.file_storage.getNode(key).mat()

    def close_file(self):
        self.file_storage.release()
        return True
