import React, { useState, useMemo, useRef, useEffect } from 'react';
import pt from 'prop-types';
import Plotly from 'plotly.js';

import ScatterPlot from '../plots/ScatterPlot';

function EmbeddingsPlot({
  onSelect,
  selectedIds,
  embeddings,
  onUnselect,
  onComputeStarted,
  onComputeFinished,
  color,
  resetIndex,
  showScale,
  colorScale,
  markerSize,
  markerOpacity,
}) {
  const [highlightedPoints, setHighlightedPoints] = useState([]);
  const scatterRef = useRef();

  const resetSelection = () => {
    Plotly.restyle(scatterRef.current.el, { selectedpoints: [null] });
  };

  const handleSelect = (eventData) => {
    if (eventData && eventData.points.length) {
      const { points } = eventData;
      onSelect(points.map((p) => p.customdata));
      setHighlightedPoints(points[0].data.selectedpoints);
    }
  };

  const handleUnselect = () => {
    setHighlightedPoints([]);
    resetSelection();
    onUnselect();
  };

  useEffect(() => {
    if (selectedIds.length) {
      onComputeStarted();
      const ids = new Set(selectedIds);
      const indices = embeddings.reduce((acc, item, index) => {
        if (ids.has(item.id)) {
          acc.push(index);
        }
        return acc;
      }, []);
      setHighlightedPoints(indices);
      onComputeFinished();
    }
  }, [selectedIds]);

  const scatterPoints = useMemo(
    () =>
      embeddings.reduce(
        (acc, { x, y, id, title }) => {
          acc.x.push(x);
          acc.y.push(y);
          acc.meta.push(id);
          if (title) {
            acc.label.push(title);
          } else {
            acc.label.push(id);
          }
          return acc;
        },
        { x: [], y: [], meta: [], label: [] }
      ),
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
      colorScale={colorScale}
      markerSize={markerSize}
      onDeselect={handleUnselect}
      onSelected={handleSelect}
      markerOpacity={markerOpacity}
      showScale={highlightedPoints.length > 0 ? false : showScale}
      layoutProps={{
        margin: { t: 20, b: 20, l: 30, r: showScale ? 100 : 20 },
      }}
    />
  );
}

EmbeddingsPlot.defaultProps = {
  color: [],
  selectedIds: [],
  embeddings: [],
  resetIndex: 0,
  markerSize: 3,
  markerOpacity: 1,
  showScale: false,
  colorScale: 'Bluered',
  onSelect: () => {},
  onUnselect: () => {},
  onComputeStarted: () => {},
  onComputeFinished: () => {},
};

EmbeddingsPlot.propTypes = {
  onSelect: pt.func,
  onUnselect: pt.func,
  showScale: pt.bool,
  resetIndex: pt.number,
  // eslint-disable-next-line react/forbid-prop-types
  selectedIds: pt.arrayOf(pt.string),
  color: pt.arrayOf(pt.number),
  onComputeStarted: pt.func,
  onComputeFinished: pt.func,
  markerOpacity: pt.number,
  colorScale: pt.string,
  markerSize: pt.number,
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
