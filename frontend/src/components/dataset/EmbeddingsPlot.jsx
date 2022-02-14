import React, { useState, useMemo, useRef, useEffect } from 'react';
import pt from 'prop-types';
import Plotly from 'plotly.js';

import ScatterPlot from '../plots/ScatterPlot';

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

  const scatterPoints = useMemo(
    () => ({
      x: embeddings.map(({ x }) => x),
      y: embeddings.map(({ y }) => y),
      meta: embeddings.map(({ id }) => ({ id })),
      label: embeddings.map(({ title }) => title),
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
    <ScatterPlot
      x={scatterPoints.x}
      y={scatterPoints.y}
      meta={scatterPoints.meta}
      highlighted={highlightedPoints}
      label={scatterPoints.label}
      innerRef={scatterRef}
      onDeselect={handleUnselect}
      onSelected={handleSelect}
    />
  );
}

EmbeddingsPlot.defaultProps = {
  filterResults: [],
  embeddings: [],
  resetIndex: 0,
  onSelect: () => {},
  onUnselect: () => {},
};

EmbeddingsPlot.propTypes = {
  onSelect: pt.func,
  onUnselect: pt.func,
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
