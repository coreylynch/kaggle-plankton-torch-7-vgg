require 'nn'
require 'image'
require 'xlua'
require 'paths'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local total_training_size = 30336
  local train_size = torch.floor(total_training_size * 0.8)
  local validation_size = total_training_size - train_size
  local test_size = 130400
  local resize = 96

  local data = torch.Tensor(total_training_size, 1, resize, resize):zero():float()
  local labels = torch.Tensor(total_training_size)

  local labels_to_index = {}
  local index_to_labels = {}

  local rorder = torch.randperm(total_training_size)

  local dir_idx = 0
  local i = 0

  print("Loading and resizing training")
  for d in paths.iterdirs('train') do
    xlua.progress(i, total_training_size)
    dir_idx = dir_idx + 1
    labels_to_index[d] = dir_idx
    index_to_labels[dir_idx] = d
    for f in paths.iterfiles('train/'..d) do
       i = i + 1
       im = image.load('train/'..d..'/'..f)
       data[rorder[i]] = image.scale(im, resize, resize)
       labels[rorder[i]] = labels_to_index[d]
    end
  end

  -- Load training / validation data 
  self.trainData = {
     data = data[{{1,train_size},{},{},{}}],
     labels = labels[{{1,train_size}}],
     size = function() return train_size end
  }
  local trainData = self.trainData

  self.validData = {
     data = data[{{train_size+1, total_training_size},{},{},{}}],
     labels = labels[{{train_size+1, total_training_size}}],
     size = function() return (validation_size) end
  }
  local validData = self.validData

  -- Load test data
  i = 0
  print("Loading and resizing test")
  test_data = torch.Tensor(test_size, 1, resize, resize):zero()
  for file in paths.iterfiles('test/') do
     xlua.progress(i, test_size)
     i = i + 1
     im = image.load('test/'..file)
     test_data[i] = image.scale(im, resize, resize)
  end

  self.testData = {
     data = test_data,
     size = function() return test_size end
  }
  local testData = self.testData
 end


function Provider:normalize()
  local trainData = self.trainData
  local testData = self.testData

  local mean = trainData.data:select(2,1):mean()
  local std = trainData.data:select(2,1):std()
  trainData.data:select(2,1):add(-mean)
  trainData.data:select(2,1):div(std)
  trainData.mean = mean
  trainData.std = std

  testData.data:select(2,1):add(-mean)
  testData.data:select(2,1):div(std)
end

