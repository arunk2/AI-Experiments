
class StringEncoder(object):
	def __init__(self):
		self.string_dict = {}
		self.string_idx = 0

	def getCode(self, string):
		if string not in self.string_dict:
			self.string_dict[string] = self.string_idx
			self.string_idx = self.string_idx + 1

		return self.string_dict[string]

	def getAllEncoding(self):
		return self.string_dict
