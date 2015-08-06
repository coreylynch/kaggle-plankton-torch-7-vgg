require 'nn'
require 'image'
require 'xlua'
require 'paths'
torch.setdefaulttensortype('torch.FloatTensor')

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local total_training_size = 30336
  local train_size = torch.floor(total_training_size * 0.8)
  local validation_size = total_training_size - train_size
  local test_size = 130400
  local resize = 64

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
       data[rorder[i]] = image.scale(im, resize, resize):float()
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
 end


function Provider:normalize()
  local trainData = self.trainData
  local validData = self.validData

  local mean = trainData.data:select(2,1):mean()
  local std = trainData.data:select(2,1):std()
  trainData.data:select(2,1):add(-mean)
  trainData.data:select(2,1):div(std)
  trainData.mean = mean
  trainData.std = std

  validData.data:select(2,1):add(-mean)
  validData.data:select(2,1):div(std)
end

