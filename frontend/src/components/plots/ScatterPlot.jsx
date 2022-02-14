import React, { useMemo } from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

import { plotColors } from '../../const';

function ScatterPlot({
  x,
  y,
  label,
  meta,
  innerRef,
  color,
  height,
  highlighted,
  dragMode,
  ...props
}) {
  const theme = useTheme();
  const { text } = theme.palette;

  const finalColor = useMemo(() => {
    if (!highlighted.length) {
      return null;
    }

    const colors = [];
    for (let i = 0; i < x.length; i += 1) {
      colors.push(plotColors.unselectedMarker);
    }

    highlighted.forEach((p) => {
      colors[p] = plotColors.selectedMarker;
    });

    return colors;
  }, [highlighted, color]);

  return (
    <Plot
      style={{
        width: '100%',
        height,
      }}
      useResizeHandler
      ref={innerRef}
      data={[
        {
          x,
          y,
          hovertext: label,
          hoverinfo: 'text',
          type: 'scatter',
          mode: 'markers',
          customdata: meta,
          unselected: {
            marker: {
              opacity: 1,
              color: plotColors.unselectedMarker,
            },
          },
          selected: {
            marker: {
              opacity: 1,
              color: plotColors.selectedMarker,
            },
          },
          marker: {
            size: 4,
            color: finalColor,
          },
        },
      ]}
      layout={{
        autosize: true,
        dragmode: dragMode,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: text.primary },
        uirevision: true,
        xaxis: { zeroline: false },
        yaxis: { zeroline: false },
        margin: { t: 20, b: 20, l: 30, r: 20 },
      }}
      {...props}
    />
  );
}

ScatterPlot.propTypes = {
  x: pt.arrayOf(pt.number).isRequired,
  y: pt.arrayOf(pt.number).isRequired,
  label: pt.arrayOf(pt.string),
  color: pt.oneOfType([pt.arrayOf(pt.number), pt.string]),
  height: pt.oneOfType([pt.number, pt.string]),
  meta: pt.arrayOf(pt.any),
  highlighted: pt.arrayOf(pt.number),
  // eslint-disable-next-line react/forbid-prop-types
  innerRef: pt.any,
  dragMode: pt.string,
};

ScatterPlot.defaultProps = {
  height: '100%',
  color: [],
  label: [],
  meta: [],
  innerRef: null,
  highlighted: [],
  dragMode: 'lasso',
};

export default ScatterPlot;
