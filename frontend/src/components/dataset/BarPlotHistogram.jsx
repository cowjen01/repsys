import React from 'react';
import pt from 'prop-types';

import { BarPlot } from '../plots';

function BarPlotHistogram({ hist, bins, height }) {
  return (
    <BarPlot
      height={height}
      layoutProps={{
        bargap: 0,
        xaxis: {
          tickfont: { size: 10 },
        },
        margin: { t: 20, b: 20, l: 30, r: 20 },
      }}
      data={[
        {
          x: hist.map((_, index) => `${bins[index]}-${bins[index + 1]}`),
          y: hist,
        },
      ]}
    />
  );
}

BarPlotHistogram.defaultProps = {
  height: 150,
};

BarPlotHistogram.propTypes = {
  hist: pt.arrayOf(pt.number).isRequired,
  bins: pt.arrayOf(pt.number).isRequired,
  height: pt.number,
};

export default BarPlotHistogram;
