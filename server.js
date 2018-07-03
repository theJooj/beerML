const config = require('./config.json')
const fetch = require('node-fetch')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

// Define a model for linear regression.
const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [2]}));
model.add(tf.layers.dense({
  inputShape: [3],
  units: 5
}))
model.add(tf.layers.dense({
  inputShape: [5],
  units: 1
}))
model.add(tf.layers.dense({
  units: 1
}))
//
// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(.06)});

// Generate data for training
const clientId = config.clientId
const clientSecret = config.clientSecret
const api = 'https://api.untappd.com/v4/'
const user = 'ppbrews'

let inputData, outputData

const encode  = (arg) => {
 return arg.split('').map(x => (x.charCodeAt(0) / 255)).reduce((a,b) => a + b, 0);
}

const testBeer = [{
  'style': encode('IPA - American'),
  'abv': 5.8,
  'ibu': 30
}]

const testingData = tf.tensor2d(testBeer.map((item) => {
  return [
    item.style,
    item.abv,
    item.ibu
  ]
}))

fetch(`${api}user/beers/${user}?client_id=${clientId}&client_secret=${clientSecret}&limit=50`)
  .then((res) => {
    return res.json()
  }).then((json => {
    inputData = tf.tensor2d(json.response.beers.items.map((item) => {
      const style = encode(item.beer.beer_style)
      return [
        style,
        item.beer.beer_abv,
        item.beer.beer_ibu
      ]
    }))
    outputData = tf.tensor2d(json.response.beers.items.map((item) => {
      return [
        item.rating_score
      ]
    }))
  })).then(() => {
    // inputData.print()
    trainModel()
  }).catch((error) => {
    console.log(error)
  })

// Train the model using the data.
const trainModel = () => {
  model.fit(inputData, outputData, {epochs: 500}).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(testingData).print();
  }).catch((error) => {
    console.log(error)
  })
}
