/* eslint-disable */
const webpack = require('webpack');
module.exports = function override(config, env) {
  config.resolve.fallback = {
    assert: require.resolve('assert'),
    stream: require.resolve('stream-browserify'),
    process: require.resolve('process')
  };
  config.plugins.push(
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer'],
    })
  );
  return config;
};
