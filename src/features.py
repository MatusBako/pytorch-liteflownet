import torch


class Features(torch.nn.Module):
	def __init__(self):
		super(Features, self).__init__()

		self.moduleOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleTwo = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleThr = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleFou = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
			torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleFiv = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

		self.moduleSix = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
			torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
		)

	def forward(self, tensorInput):
		tensorOne = self.moduleOne(tensorInput)
		tensorTwo = self.moduleTwo(tensorOne)
		tensorThr = self.moduleThr(tensorTwo)
		tensorFou = self.moduleFou(tensorThr)
		tensorFiv = self.moduleFiv(tensorFou)
		tensorSix = self.moduleSix(tensorFiv)

		return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]