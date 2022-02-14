import React, { useState, useMemo, useRef, useEffect } from 'react';
import pt from 'prop-types';
import { Paper } from '@mui/material';
import Plotly from 'plotly.js';

import ScatterPlot from './ScatterPlot';
import { plotColors } from '../../const';

function EmbeddingsPlot({ onSelect, filterResults, embeddings, onUnselect, resetIndex }) {
  const [highlightedPoints, setHighlightedPoints] = useState([]);
  const scatterRef = useRef();

  const resetSelection = () => {
    Plotly.restyle(scatterRef.current.el, { selectedpoints: [null] });
  };

  const handleSelect = (eventData) => {
    if (eventData && eventData.points.length) {
      const { points } = eventData;
      onSelect(points.map((p) => p.customdata.id));
      setHighlightedPoints(points[0].data.selectedpoints);
    }
  };

  const handleUnselect = () => {
    setHighlightedPoints([]);
    resetSelection();
    onUnselect();
  };

  useEffect(() => {
    if (filterResults.length) {
      const indices = embeddings.reduce((acc, item, index) => {
        if (filterResults.includes(item.id)) {
          acc.push(index);
        }
        return acc;
      }, []);
      setHighlightedPoints(indices);
    }
  }, [filterResults]);

  const scatterColors = useMemo(() => {
    if (!highlightedPoints.length) {
      return plotColors.selectedMarker;
    }

    const colors = [];
    for (let i = 0; i < embeddings.length; i += 1) {
      colors.push(plotColors.unselectedMarker);
    }

    highlightedPoints.forEach((p) => {
      colors[p] = plotColors.selectedMarker;
    });

    return colors;
  }, [highlightedPoints]);

  const scatterPoints = useMemo(
    () => ({
      x: embeddings.map(({ x }) => x),
      y: embeddings.map(({ y }) => y),
      meta: embeddings.map(({ id }) => ({ id })),
      label: embeddings.map(({ label }) => label),
    }),
    [embeddings]
  );

  useEffect(() => {
    if (resetIndex > 0) {
      setHighlightedPoints([]);
      resetSelection();
    }
  }, [resetIndex]);

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <ScatterPlot
        x={scatterPoints.x}
        y={scatterPoints.y}
        meta={scatterPoints.meta}
        color={scatterColors}
        label={scatterPoints.label}
        innerRef={scatterRef}
        onDeselect={handleUnselect}
        onSelected={handleSelect}
      />
    </Paper>
  );
}

EmbeddingsPlot.defaultProps = {
  filterResults: [],
  embeddings: [],
  resetIndex: 0,
};

EmbeddingsPlot.propTypes = {
  onSelect: pt.func.isRequired,
  onUnselect: pt.func.isRequired,
  resetIndex: pt.number,
  // eslint-disable-next-line react/forbid-prop-types
  filterResults: pt.arrayOf(pt.number),
  embeddings: pt.arrayOf(
    pt.shape({
      title: pt.string,
      x: pt.number,
      y: pt.number,
      id: pt.number,
    })
  ),
};

export default EmbeddingsPlot;
