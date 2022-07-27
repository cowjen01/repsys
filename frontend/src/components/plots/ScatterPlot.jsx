import React, { useMemo } from 'react';
import pt from 'prop-types';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

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
  showScale,
  colorScale,
  layoutProps,
  markerSize,
  markerOpacity,
  unselectedColor,
  selectedColor,
  ...props
}) {
  const theme = useTheme();

  const gridcolor = theme.palette.mode === 'dark' ? theme.palette.divider : null;

  const defaultColors = useMemo(() => {
    const colors = [];
    for (let i = 0; i < x.length; i += 1) {
      colors.push(unselectedColor);
    }
    return colors;
  }, [x.length]);

  const finalColor = useMemo(() => {
    if (!highlighted.length && !color.length) {
      return selectedColor;
    }

    if (!highlighted.length && color.length) {
      return color;
    }

    const colors = [...defaultColors];
    highlighted.forEach((p) => {
      colors[p] = selectedColor;
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
          type: 'scattergl',
          mode: 'markers',
          customdata: meta,
          unselected: {
            marker: {
              opacity: markerOpacity,
              color: unselectedColor,
            },
          },
          selected: {
            marker: {
              opacity: markerOpacity,
              color: selectedColor,
            },
          },
          marker: {
            size: markerSize,
            color: finalColor,
            showscale: showScale,
            colorscale: colorScale,
            opacity: markerOpacity,
            cmin: -1,
            cmax: 1,
          },
        },
      ]}
      layout={{
        autosize: true,
        dragmode: dragMode,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: theme.palette.text.primary },
        uirevision: true,
        xaxis: {
          zeroline: false,
          gridcolor,
        },
        yaxis: {
          zeroline: false,
          gridcolor,
          scaleanchor: 'x',
          scaleratio: 1,
        },
        margin: { t: 20, b: 20, l: 30, r: 20 },
        ...layoutProps,
      }}
      {...props}
    />
  );
}

ScatterPlot.propTypes = {
  x: pt.arrayOf(pt.number).isRequired,
  y: pt.arrayOf(pt.number).isRequired,
  label: pt.arrayOf(pt.oneOfType([pt.string, pt.number])),
  color: pt.oneOfType([pt.arrayOf(pt.number), pt.string]),
  height: pt.oneOfType([pt.number, pt.string]),
  meta: pt.arrayOf(pt.any),
  highlighted: pt.arrayOf(pt.number),
  innerRef: pt.any,
  dragMode: pt.string,
  markerOpacity: pt.number,
  markerSize: pt.number,
  showScale: pt.bool,
  layoutProps: pt.any,
  colorScale: pt.string,
  unselectedColor: pt.string,
  selectedColor: pt.string,
};

ScatterPlot.defaultProps = {
  height: '100%',
  color: [],
  label: [],
  meta: [],
  showScale: false,
  innerRef: null,
  highlighted: [],
  markerOpacity: 1,
  markerSize: 2,
  dragMode: 'lasso',
  layoutProps: {},
  colorScale: 'Bluered',
  unselectedColor: '#dcdcdc',
  selectedColor: '#0613ff',
};

export default ScatterPlot;
