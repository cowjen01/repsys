import React, { useState, useMemo, useRef, useEffect } from 'react';
import pt from 'prop-types';
import Plotly from 'plotly.js';

import ScatterPlot from '../plots/ScatterPlot';

function EmbeddingsPlot({
  onSelect,
  filterResults,
  embeddings,
  onUnselect,
  color,
  resetIndex,
  showScale,
}) {
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
      color={color}
      highlighted={highlightedPoints}
      label={scatterPoints.label}
      innerRef={scatterRef}
      onDeselect={handleUnselect}
      onSelected={handleSelect}
      showScale={highlightedPoints.length > 0 ? false : showScale}
      layoutProps={{
        margin: { t: 20, b: 20, l: 30, r: showScale ? 100 : 20 },
      }}
    />
  );
}

EmbeddingsPlot.defaultProps = {
  color: [],
  filterResults: [],
  embeddings: [],
  resetIndex: 0,
  showScale: false,
  onSelect: () => {},
  onUnselect: () => {},
};

EmbeddingsPlot.propTypes = {
  onSelect: pt.func,
  onUnselect: pt.func,
  showScale: pt.bool,
  resetIndex: pt.number,
  // eslint-disable-next-line react/forbid-prop-types
  filterResults: pt.arrayOf(pt.number),
  color: pt.arrayOf(pt.number),
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
