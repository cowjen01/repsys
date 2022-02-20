import React from 'react';
import pt from 'prop-types';

import { BarPlot } from '../plots';

function BarPlotHistogram({ values, bins }) {
  return (
    <BarPlot
      height={150}
      width={300}
      layoutProps={{
        bargap: 0,
        xaxis: {
          tickfont: { size: 10 },
        },
        yaxis: {
          visible: false,
        },
        margin: { t: 20, b: 20, l: 0, r: 0 },
      }}
      data={[
        {
          x: values.map((_, index) => `${bins[index]}-${bins[index + 1]}`),
          y: values,
        },
      ]}
    />
  );
}

BarPlotHistogram.propTypes = {
  values: pt.arrayOf(pt.number).isRequired,
  bins: pt.arrayOf(pt.number).isRequired,
};

export default BarPlotHistogram;
