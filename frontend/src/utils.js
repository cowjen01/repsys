/* eslint-disable no-bitwise */
function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isObject(item) {
  return item && typeof item === 'object' && !Array.isArray(item);
}

function mergeDeep(target, ...sources) {
  if (!sources.length) return target;
  const source = sources.shift();

  if (isObject(target) && isObject(source)) {
    // eslint-disable-next-line no-restricted-syntax
    for (const key in source) {
      if (isObject(source[key])) {
        if (!target[key]) Object.assign(target, { [key]: {} });
        mergeDeep(target[key], source[key]);
      } else {
        Object.assign(target, { [key]: source[key] });
      }
    }
  }

  return mergeDeep(target, ...sources);
}

function sliceIdentifier(id, limit = 20) {
  if (!id) return '';
  const str = id.toString();
  if (str.length < limit) return id;
  return `${str.slice(0, limit)}...`;
}

function generateUID() {
  let firstPart = (Math.random() * 46656) | 0;
  let secondPart = (Math.random() * 46656) | 0;
  firstPart = `000${firstPart.toString(36)}`.slice(-3);
  secondPart = `000${secondPart.toString(36)}`.slice(-3);
  return firstPart + secondPart;
}

export { sleep, capitalize, mergeDeep, sliceIdentifier, generateUID };
