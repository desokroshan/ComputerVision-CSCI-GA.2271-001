require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require 'gnuplot'
--[[
--  Try to Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 48, 48
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
--local dbg = require 'debugger.lua'
--dbg()
torch.manualSeed(opt.manualSeed)
if opt.cuda == "true" then
  require 'cunn'
  require 'cudnn' -- faster convolutions
  cutorch.manualSeedAll(opt.manualSeed)
end

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

-- Add some more transformations
function rotate(img)
    return image.rotate(img, torch.random(-10,10)*math.pi/180)
end

function translate(img)
    return image.translate(img, torch.random(-2,2), torch.random(-2,2))
end

function horizontalFlip(img)
    return image.hflip(img)
end

function rgb2yuv(img)
    return image.rgb2yuv(img)
end
function localNormalization(img)
    return nn.SpatialContrastiveNormalization(3, image.gaussian1D(3)):forward(img)
end

function globalNormalization(img)
    mean = img:mean()
    std  = img:std()
    img:add(-mean)
    img:div(std)
    return img
end

function identity(img)
    return img
end
--[[
-- Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function preprocessInput(inp)
    f = tnt.transform.compose{
	[1] = resize,
	[2] = rgb2yuv,
	[3] = globalNormalization,
	[4] = localNormalization,
    }
    return f(inp)
end


function getTrainSample(dataset, idx, transformation)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    img = image.load(DATA_PATH .. '/train_images/'..file)
    --img = image.crop(img, r[5], r[6], r[7], r[8])
    preprocessed_image = preprocessInput(img)
    return transformation(preprocessed_image)
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    --img = image.crop(image.load(file), r[4], r[5], r[6], r[7])
    return preprocessInput(image.load(file))
end

function getIterator(dataset)
    --[[
    -- Hint:  Use ParallelIterator for using multiple CPU cores
    --]]
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

local loadFunc = function(transformation)
                    return function(idx)
                return {
                    input =  getTrainSample(trainData, idx, transformation),
                    target = getTrainLabel(trainData, idx)
                }
            end
        end
preprocessedData = tnt.ListDataset{list = torch.range(1, trainData:size(1)):long(),load = loadFunc(identity)}
rotatedData = tnt.ListDataset{list = torch.range(1, trainData:size(1)):long(),load = loadFunc(rotate)}
translatedData = tnt.ListDataset{list = torch.range(1, trainData:size(1)):long(),load = loadFunc(translate)}
trainDataset  = tnt.SplitDataset{partitions = {train=0.9, val=0.1},initialpartition = 'train', dataset = tnt.ShuffleDataset{dataset = tnt.ConcatDataset{datasets={preprocessedData, rotatedData, translatedData}}}}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}


--[[
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model)
local criterion = nn.CrossEntropyCriterion()
if opt.cuda == "true" then
  print("Inside Cuda2")
  model = model:cuda()
  criterion = criterion:cuda()
end
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

-- print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

--[[
-- Hint:  Use onSample function to convert to 
--        cuda tensor for using GPU
--]]
if opt.cuda == "true" then
  print("Inside Cuda3")
  local input = torch.CudaTensor()
  local target = torch.CudaTensor()
  engine.hooks.onSample = function(state)
    input:resize(state.sample.input:size()):copy(state.sample.input)
    target:resize(state.sample.target:size()):copy(state.sample.target)
    state.sample.input  = input
    state.sample.target = target
  end
end

local errors_train = {}
local errors_val = {}
local loss_train = {}
local loss_val = {}
local bool = 0

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
    if bool%2 == 0 then
        errors_train[#errors_train + 1] = clerr:value{k = 1}
        loss_train[#loss_train + 1] = meter:value()
    else
        errors_val[#errors_val + 1] = clerr:value{k = 1}
        loss_val[#loss_val + 1] = meter:value()
    end
    bool = bool + 1
end

local epoch = 1

while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

if opt.cuda == "true" then
  local input = torch.CudaTensor()
  local sampleId = torch.CudaTensor()
  engine.hooks.onSample = function(state)
    input:resize(state.sample.input:size()):copy(state.sample.input)
    sampleId:resize(state.sample.sampleId:size()):copy(state.sample.sampleId)
    state.sample.input  = input
    state.sample.sampleId = sampleId
  end
end

gnuplot.pngfigure('train_val_error_vgg_pp_da_c48.png')
gnuplot.plot({torch.range(1, #errors_train),torch.Tensor(errors_train),'-'},{torch.range(1, #errors_val),torch.Tensor(errors_val),'-'})
gnuplot.xlabel('EPOCHS')
gnuplot.ylabel('TRAIN_VAL_ERROR')
gnuplot.plotflush()


gnuplot.pngfigure('Train_val_loss_vgg_pp_da_c48.png')
gnuplot.plot({torch.range(1, #loss_train),torch.Tensor(loss_train),'-'},{torch.range(1, #loss_val),torch.Tensor(loss_val),'-'})
gnuplot.xlabel('NUM_EPOCHS')
gnuplot.ylabel('TRAIN_VAL_LOSS')
gnuplot.plotflush()

local submission = assert(io.open(opt.logDir .. "/submission_vgg_pp_da_c48.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

model:clearState()
torch.save("trained_vgg_pp_da_c48.t7",model)
print("The End!")
