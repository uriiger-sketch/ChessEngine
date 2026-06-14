function ChessNN

    % Global variables for neural network
    global net useNN;
    net = []; useNN = false;
    netModified = false;  % track unsaved changes to the network

    % Engine parameters (tuned for higher performance)
    maxDepth   = 12;       % maximum search depth for AI (iterative deepening limit)
    baseTime   = 5;        % default AI think time per move (seconds)
    gammaNet   = 0.95;      % weight of neural network evaluation in total score
    MATE_SCORE = 5000;    % score for checkmate positions (for alpha-beta evaluation)
    
    % Initialize game state and GUI components
    state = initState();   % game state structure (board, castling rights, etc.)
    currentPlayer = 1;     % 1 = human (White), 2 = AI (Black)
    
    % GUI layout parameters
    boardSize = 480;
    ctrlWidth = 300;
    ctrlColor = [0.92 0.93 0.95];
    colLight  = [0.93 0.94 0.95];
    colDark   = [0.78 0.83 0.86];
    highlightColor = [1.0 0.92 0.45];  % highlight color for last move
    
    % Create main GUI figure (non-resizable)
    fig = figure('Name','ChessNN','NumberTitle','off','MenuBar','none','Toolbar','none',...
                 'Resize','off','Color',ctrlColor,...
                 'Position',[100 100 boardSize+ctrlWidth boardSize]);
    % Board panel
    boardPanel = uipanel(fig, 'Units','pixels', 'Position',[0 0 boardSize boardSize],...
                          'BackgroundColor',colDark, 'BorderType','none');
    % Control panel (side bar)
    ctrlPanel = uipanel(fig, 'Units','pixels', 'Position',[boardSize 0 ctrlWidth boardSize],...
                         'BackgroundColor',ctrlColor, 'BorderType','none');
    % Status text box
    statusBox = uicontrol(ctrlPanel, 'Style','text', 'Units','normalized', ...
        'Position',[0.05 0.17 0.90 0.10], 'HorizontalAlignment','left', ...
        'BackgroundColor',ctrlColor, 'FontSize',9, 'String','Ready.');
    % Captured pieces display panels
    capPanelWhite = uipanel(ctrlPanel, 'Units','pixels', 'Position',[(ctrlWidth-288)/2, boardSize-325, 288, 20],...
                             'BackgroundColor',ctrlColor, 'BorderType','none');
    capPanelBlack = uipanel(ctrlPanel, 'Units','pixels', 'Position',[(ctrlWidth-288)/2, boardSize-350, 288, 20],...
                             'BackgroundColor',ctrlColor, 'BorderType','none');
    capLblWhite = gobjects(1,16);
    capLblBlack = gobjects(1,16);
    for k = 1:16
        capLblWhite(k) = uicontrol(capPanelWhite, 'Style','text', 'String',' ', 'FontSize',14,...
            'Units','pixels', 'Position',[(k-1)*18 0 18 20], 'HorizontalAlignment','center',...
            'BackgroundColor',ctrlColor, 'ForegroundColor',[0.2 0.2 0.2]);
        capLblBlack(k) = uicontrol(capPanelBlack, 'Style','text', 'String',' ', 'FontSize',14,...
            'Units','pixels', 'Position',[(k-1)*18 0 18 20], 'HorizontalAlignment','center',...
            'BackgroundColor',ctrlColor, 'ForegroundColor',[0.2 0.2 0.2]);
    end
    
    % Create 8x8 grid of buttons for the chessboard squares
    boardBtns = gobjects(8,8);
    for r = 1:8
        for c = 1:8
            boardBtns(r,c) = uicontrol(boardPanel, 'Style','pushbutton', 'Units','normalized', ...
                'Position',[(c-1)/8, 1 - r/8, 1/8, 1/8], 'FontSize',24, 'FontWeight','bold', ...
                'BackgroundColor',squareColor(mod(r+c,2)==0), 'UserData',[r c], 'Callback',@onBoardClick);
        end
    end
    
    drawBoard();      % draw initial pieces
    updateCaptured(); % clear captured pieces display
    
    % Control buttons
    uicontrol(ctrlPanel, 'Style','pushbutton', 'String','New Game', ...
              'Units','normalized', 'FontSize',10, 'BackgroundColor',[0.85 0.87 0.89],...
              'Position',[0.10 0.90 0.80 0.06], 'Callback',@newGame);
    uicontrol(ctrlPanel, 'Style','pushbutton', 'String','Save Network', ...
              'Units','normalized', 'FontSize',10, 'BackgroundColor',[0.85 0.87 0.89],...
              'Position',[0.10 0.82 0.80 0.06], 'Callback',@saveNetwork);
    uicontrol(ctrlPanel, 'Style','pushbutton', 'String','Load Network', ...
              'Units','normalized', 'FontSize',10, 'BackgroundColor',[0.85 0.87 0.89],...
              'Position',[0.10 0.74 0.80 0.06], 'Callback',@loadNetwork);
    uicontrol(ctrlPanel, 'Style','pushbutton', 'String','Train from PGN', ...
              'Units','normalized', 'FontSize',10, 'BackgroundColor',[0.85 0.87 0.89],...
              'Position',[0.10 0.66 0.80 0.06], 'Callback',@trainNetwork);
    
    % AI think time slider and label
    uicontrol(ctrlPanel, 'Style','text', 'String','AI Think Time:', 'Units','normalized', ...
              'Position',[0.10 0.53 0.50 0.03], 'BackgroundColor',ctrlColor, ...
              'FontWeight','bold', 'HorizontalAlignment','left');
    thinkTimes = [2, 5, 10];  % possible time limits (seconds)
    thinkSlider = uicontrol(ctrlPanel, 'Style','slider', 'Min',1, 'Max',3, 'Value',2, 'SliderStep',[0.5 0.5],...
              'Units','normalized', 'Position',[0.10 0.48 0.50 0.03], 'Callback',@changeThinkTime);
    thinkLabel = uicontrol(ctrlPanel, 'Style','text', 'String','5 s', 'Units','normalized', ...
              'Position',[0.62 0.48 0.25 0.03], 'BackgroundColor',ctrlColor, 'HorizontalAlignment','left');
    baseTime = thinkTimes(round(thinkSlider.Value));
    thinkLabel.String = sprintf('%.0f s', baseTime);
    
    %% GUI Callback Functions %%
    function drawBoard()
        % Redraw the GUI board buttons to match the current state.board
        for rr = 1:8
            for cc = 1:8
                boardBtns(rr,cc).String = pieceGlyph(state.board(rr,cc));
                boardBtns(rr,cc).BackgroundColor = squareColor(mod(rr+cc,2)==0);
            end
        end
    end

    function updateCaptured()
        % Update the captured pieces display for both White and Black
        for k = 1:16
            capLblWhite(k).String = ' ';
            capLblBlack(k).String = ' ';
            if k <= numel(state.capturedWhite)
                capLblWhite(k).String = pieceGlyph(charToCode(state.capturedWhite{k}));
            end
            if k <= numel(state.capturedBlack)
                capLblBlack(k).String = pieceGlyph(charToCode(state.capturedBlack{k}));
            end
        end
    end

    function newGame(~,~)
        % Start a new game: reset state and update GUI
        state = initState();
        currentPlayer = 1;
        drawBoard();
        updateCaptured();
        statusBox.String = 'New game started. Your move.';
    end

    function saveNetwork(~,~)
        % Save the current neural network to a .mat file
        if isempty(net)
            msgbox('No neural network to save.','Notice','warn'); 
            return;
        end
        [file, path] = uiputfile('*.mat', 'Save Neural Network');
        if isequal(file,0), return; end
        fname = fullfile(path, file);
        try
            netClean = removeHandles(net);
            net = netClean; %#ok<NASGU>
            save(fname, 'net');
            netModified = false;
            statusBox.String = ['Network saved to ' fname];
        catch e
            errordlg(['Error saving network: ' e.message], 'Save Error');
        end
    end

    function loadNetwork(~,~)
        % Load a neural network from a .mat file
        if ~isempty(net) && netModified
            % Prompt to save unsaved network first
            choice = questdlg('Current network has unsaved changes. Save it before loading a new one?', ...
                               'Unsaved Network', 'Yes','No','Cancel','Cancel');
            if strcmp(choice, 'Cancel'), return; end
            if strcmp(choice, 'Yes'), saveNetwork(); end
        end
        [file, path] = uigetfile('*.mat', 'Load Neural Network');
        if isequal(file,0), return; end
        S = load(fullfile(path, file));
        if isfield(S,'net')
            net = removeHandles(S.net);
        else
            errordlg('MAT-file does not contain a valid network.', 'Load Error');
            return;
        end
        useNN = true;
        netModified = false;
        statusBox.String = sprintf('Neural network loaded (%d layers).', numel(net.layers));
    end

    %-----------------------------------------------------------------------
% 2) STREAM / MINI‑BATCH TRAINING
%-----------------------------------------------------------------------
function trainNetwork(~,~)
%   Reads *any* number of PGN files (even multi‑GB); trains the
%   network a few epochs at a time on fixed‑size mini‑batches so
%   working RAM stays modest and never builds a giant matrix.

    [files,path] = uigetfile({'*.pgn','PGN files'}, ...
                             'Select PGN file(s)','MultiSelect','on');
    if isequal(files,0),  return;  end
    if ischar(files),     files = {files}; end           %#ok<*ISCHAR>

    %---------------------------------------------------- network init
    if isempty(net)
        net = createNetwork();   % new
    end
    useNN       = true;
    netModified = true;

    %---------------------------------------------------- parameters
    BATCH_SIZE   = 5000;   % <- change if you like
    BATCH_EPOCHS = 3;      % epochs per mini‑batch update

    statusBox.String = 'Scanning PGN…';   drawnow;

    % persistent mapminmax structs (filled on first batch)
    Xnorm = state.Xnorm;
    Ynorm = state.Ynorm;

    % empty batch buffers
    Xbuf  = zeros(768,BATCH_SIZE,'single');    % single → ¼ memory
    Ybuf  = zeros(1,BATCH_SIZE, 'single');
    nInBatch = 0;
    totalPos = 0;
    batchCnt = 0;

    %---------------------------------------------------- helper
    function feedBatch()
        if nInBatch==0,  return, end
        batchCnt  = batchCnt+1;

        Xi = Xbuf(:,1:nInBatch);
        Yi = Ybuf(:,1:nInBatch);

        % map‑min‑max: first batch defines ranges, later batches reuse
        if isempty(Xnorm)
            [Xi, Xnorm] = mapminmax(Xi);
            [Yi, Ynorm] = mapminmax(Yi);
            state.Xnorm = Xnorm;   %#ok<*NASGU>
            state.Ynorm = Ynorm;
        else
            Xi = mapminmax('apply',Xi,Xnorm);
            Yi = mapminmax('apply',Yi,Ynorm);
        end

        % tiny number of epochs keeps each step cheap
        net.trainParam.epochs = BATCH_EPOCHS;

        % **incremental** learning – keeps current weights
        net = train(net,Xi,Yi,'showWindow','off','showCommandLine','off');

        % reset buffer
        nInBatch = 0;
        if batchCnt==1 || mod(batchCnt,10)==0
            statusBox.String = sprintf('   trained %d batches, %d positions…', ...
                                       batchCnt,totalPos);
            drawnow;
        end
    end

    %---------------------------------------------------- MAIN LOOP : read → sample → buffer
    for f = 1:numel(files)
        raw = fileread(fullfile(path,files{f}));
        games = splitPGN(raw);                % existing util
        for g = 1:numel(games)
            [Xg,Yg] = parsePGNGame(games{g}); % existing util
            if isempty(Xg),  continue,  end
            k = size(Xg,2);
            for i = 1:k
                nInBatch        = nInBatch + 1;
                totalPos        = totalPos + 1;
                Xbuf(:,nInBatch)= single(Xg(:,i));
                Ybuf(1,nInBatch)= single(Yg(1,i));

                if nInBatch == BATCH_SIZE
                    feedBatch();
                end
            end
        end
        statusBox.String = sprintf('Parsed file %d / %d – %d games', ...
                                   f,numel(files),numel(games));
        drawnow;
    end

    % train on the final, possibly smaller, batch
    feedBatch();

    statusBox.String = sprintf('✅ Incremental training finished. %d total positions, %d batches.', ...
                               totalPos,batchCnt);
    drawnow;
end


    function changeThinkTime(src, ~)
        % Adjust the AI think time based on slider value
        idx = round(src.Value);
        src.Value = idx;
        baseTime = thinkTimes(idx);
        thinkLabel.String = sprintf('%.0f s', baseTime);
    end

    function onBoardClick(src, ~)
        % Handle a click on the board by the human (White) player
        if currentPlayer ~= 1
            return;  % ignore clicks if it's not White's turn
        end
        pos = src.UserData;
        r = pos(1); c = pos(2);
        if isempty(state.selected)
            % No piece selected yet: select this square if it has a White piece
            if state.board(r,c) > 0
                state.selected = [r c];
                src.BackgroundColor = [1 1 0.6];
            end
        else
            % A piece is already selected; interpret click as target square
            from = state.selected;
            to = [r c];
            if isequal(to, from)
                % Deselect if same square clicked again
                boardBtns(from(1), from(2)).BackgroundColor = squareColor(mod(sum(from),2)==0);
                state.selected = [];
                return;
            end
            % Check if the move is legal for the selected piece
            moves = generateMoves(state, 'white');
            idx = find(arrayfun(@(m) isequal(m.from,from) && isequal(m.to,to), moves));
            if isempty(idx)
                return; % invalid move (ignore)
            end
            move = moves(idx);
            % Make the move and update state
            state = makeMove(state, move);
            state.selected = [];
            highlightLastMove(move.from, move.to);
            drawBoard();
            updateCaptured();
            % Check if Black has any moves (game over condition after White's move)
            [gameOver, isMate] = isGameOver(state, 'black');
            if gameOver
                if isMate
                    msgbox('Checkmate! White wins.', 'Game Over');
                else
                    msgbox('Stalemate.', 'Game Over');
                end
                currentPlayer = 0;
                return;
            end
            % Switch turn to Black (AI)
            currentPlayer = 2;
            statusBox.String = 'Computer is thinking...'; drawnow;
            pause(0.1);
            makeAIMove();
        end
    end

    function makeAIMove()
        % Compute and execute the AI (Black) move
        rootKey = zobristKey(state.board);
        bestMove = [];
        t0 = tic;
        % Iterative deepening search up to maxDepth or time limit
        for depth = 1:maxDepth
            [mv, ~] = search(state, rootKey, depth, -Inf, Inf, 'black', 0);
            if ~isempty(mv)
                bestMove = mv;
            end
            if toc(t0) >= baseTime
                break;
            end
        end
        if isempty(bestMove)
            % No legal moves: Black is checkmated or stalemated
            [gameOver, isMate] = isGameOver(state, 'black');
            if gameOver
                if isMate
                    msgbox('Checkmate! White wins.', 'Game Over');
                else
                    msgbox('Stalemate.', 'Game Over');
                end
            end
            currentPlayer = 0;
            return;
        end
        % Apply the chosen move
        state = makeMove(state, bestMove);
        highlightLastMove(bestMove.from, bestMove.to);
        drawBoard();
        updateCaptured();
        % If the White king was captured by this move (game over)
        if bestMove.captured == 6
            msgbox('Black wins by capturing the White King!', 'Game Over');
            currentPlayer = 0;
            return;
        end
        % Switch back to White's turn
        currentPlayer = 1;
        statusBox.String = 'Your move.';
        % Check if White has any moves (checkmate after Black's move)
        [gameOver, isMate] = isGameOver(state, 'white');
        if gameOver
            if isMate
                msgbox('Checkmate! Black wins.', 'Game Over');
            else
                msgbox('Stalemate.', 'Game Over');
            end
            currentPlayer = 0;
        end
    end

    function highlightLastMove(fromSq, toSq)
        % Highlight the last move on the board (both origin and destination)
        if isfield(state, 'lastMove') && ~isempty(state.lastMove)
            % Remove previous highlights
            prevFrom = state.lastMove.from;
            prevTo   = state.lastMove.to;
            boardBtns(prevFrom(1), prevFrom(2)).BackgroundColor = squareColor(mod(sum(prevFrom),2)==0);
            boardBtns(prevTo(1), prevTo(2)).BackgroundColor     = squareColor(mod(sum(prevTo),2)==0);
        end
        boardBtns(fromSq(1), fromSq(2)).BackgroundColor = highlightColor;
        boardBtns(toSq(1), toSq(2)).BackgroundColor     = highlightColor;
    end

    %% Core Game Functions %%
    function st = initState()
        % Initialize a new game state with the standard chess starting position
        % Piece codes: pawn=1, knight=2, bishop=3, rook=4, queen=5, king=6 (White positive, Black negative)
        st.board = [ ...
            -4 -2 -3 -5 -6 -3 -2 -4;   % 8th rank (Black major pieces)
            -1 -1 -1 -1 -1 -1 -1 -1;   % 7th rank (Black pawns)
             0  0  0  0  0  0  0  0;   % 6th rank
             0  0  0  0  0  0  0  0;   % 5th rank
             0  0  0  0  0  0  0  0;   % 4th rank
             0  0  0  0  0  0  0  0;   % 3rd rank
             1  1  1  1  1  1  1  1;   % 2nd rank (White pawns)
             4  2  3  5  6  3  2  4    % 1st rank (White major pieces)
        ];
        st.wKc = true; st.wQc = true;  % White king-side and queen-side castling rights
        st.bKc = true; st.bQc = true;  % Black king-side and queen-side castling rights
        st.capturedWhite = {};         % list of White pieces captured (White pieces captured by Black)
        st.capturedBlack = {};         % list of Black pieces captured (Black pieces captured by White)
        st.lastMove = [];              % last move made (for en passant and highlighting)
        st.selected = [];              % selected square (for GUI input)
        st.Xnorm = []; st.Ynorm = [];  % normalization settings for NN inputs/outputs (set during training)
    end

    function moves = generateMoves(st, side)
        % Generate all legal moves for the given side ('white' or 'black')
        moves = struct('from',{}, 'to',{}, 'piece',{}, 'captured',{});
        friendSign = 1;
        if strcmp(side,'black'), friendSign = -1; end
        enemySign = -friendSign;
        board = st.board;
        % Determine en passant target square if applicable
        epTarget = [];
        if ~isempty(st.lastMove) && abs(st.lastMove.piece) == 1 && ...
           abs(st.lastMove.from(1) - st.lastMove.to(1)) == 2
            % Last move was a pawn double-step; en passant capture possible on the square passed over
            epTarget = [(st.lastMove.from(1) + st.lastMove.to(1)) / 2, st.lastMove.to(2)];
        end
        % Find all pieces belonging to the side to move
        [pieceRows, pieceCols] = find(sign(board) == friendSign);
        for idx = 1:numel(pieceRows)
            r = pieceRows(idx);
            c = pieceCols(idx);
            p = board(r,c);
            pieceType = friendSign * p;  % positive piece type (1-6) for the moving side
            switch pieceType
                case 1  % Pawn moves
                    if friendSign == 1
                        forward = -1; startRow = 7; epRow = 4; promotionRow = 1; epPawnCode = -1;
                    else
                        forward = 1; startRow = 2; epRow = 5; promotionRow = 8; epPawnCode = 1;
                    end
                    nextR = r + forward;
                    if nextR >= 1 && nextR <= 8
                        % Move forward (one square)
                        if board(nextR, c) == 0
                            addMove([r c], [nextR c], p, 0);
                            % Double move from starting row
                            if r == startRow && board(r + 2*forward, c) == 0
                                addMove([r c], [r + 2*forward, c], p, 0);
                            end
                        end
                        % Captures (including en passant)
                        for dc = [-1, 1]
                            cc = c + dc;
                            if cc < 1 || cc > 8, continue; end
                            % Normal diagonal capture
                            if board(nextR, cc) * friendSign < 0
                                addMove([r c], [nextR cc], p, board(nextR, cc));
                            % En passant capture
                            elseif ~isempty(epTarget) && r == epRow && isequal([nextR cc], epTarget)
                                addMove([r c], epTarget, p, epPawnCode);
                            end
                        end
                    end
                case 2  % Knight moves
                    knightOffsets = [1 2; 1 -2; -1 2; -1 -2; 2 1; 2 -1; -2 1; -2 -1];
                    for k = 1:size(knightOffsets,1)
                        nr = r + knightOffsets(k,1);
                        nc = c + knightOffsets(k,2);
                        if nr < 1 || nr > 8 || nc < 1 || nc > 8, continue; end
                        if sign(board(nr, nc)) ~= friendSign
                            addMove([r c], [nr nc], p, board(nr, nc));
                        end
                    end
                case 3  % Bishop moves (diagonals)
                    directions = [1 1; 1 -1; -1 1; -1 -1];
                    for d = 1:4
                        dr = directions(d,1);
                        dc = directions(d,2);
                        nr = r + dr; nc = c + dc;
                        while nr >= 1 && nr <= 8 && nc >= 1 && nc <= 8
                            if board(nr, nc) == 0
                                addMove([r c], [nr nc], p, 0);
                            else
                                if sign(board(nr, nc)) == enemySign
                                    addMove([r c], [nr nc], p, board(nr, nc));
                                end
                                break;  % stop at first piece encountered
                            end
                            nr = nr + dr; nc = nc + dc;
                        end
                    end
                case 4  % Rook moves (straight lines)
                    directions = [1 0; -1 0; 0 1; 0 -1];
                    for d = 1:4
                        dr = directions(d,1);
                        dc = directions(d,2);
                        nr = r + dr; nc = c + dc;
                        while nr >= 1 && nr <= 8 && nc >= 1 && nc <= 8
                            if board(nr, nc) == 0
                                addMove([r c], [nr nc], p, 0);
                            else
                                if sign(board(nr, nc)) == enemySign
                                    addMove([r c], [nr nc], p, board(nr, nc));
                                end
                                break;
                            end
                            nr = nr + dr; nc = nc + dc;
                        end
                    end
                case 5  % Queen moves (rook + bishop directions)
                    directions = [1 0; -1 0; 0 1; 0 -1; 1 1; 1 -1; -1 1; -1 -1];
                    for d = 1:8
                        dr = directions(d,1);
                        dc = directions(d,2);
                        nr = r + dr; nc = c + dc;
                        while nr >= 1 && nr <= 8 && nc >= 1 && nc <= 8
                            if board(nr, nc) == 0
                                addMove([r c], [nr nc], p, 0);
                            else
                                if sign(board(nr, nc)) == enemySign
                                    addMove([r c], [nr nc], p, board(nr, nc));
                                end
                                break;
                            end
                            nr = nr + dr; nc = nc + dc;
                        end
                    end
                case 6  % King moves (adjacent squares + castling)
                    kingOffsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
                    for k = 1:size(kingOffsets,1)
                        nr = r + kingOffsets(k,1);
                        nc = c + kingOffsets(k,2);
                        if nr < 1 || nr > 8 || nc < 1 || nc > 8, continue; end
                        if sign(board(nr, nc)) ~= friendSign
                            addMove([r c], [nr nc], p, board(nr, nc));
                        end
                    end
                    % Castling moves (if king is on original square and path is clear and safe)
                    if friendSign == 1 && r == 8 && c == 5
                        % White king on e1
                        if st.wKc && board(8,6)==0 && board(8,7)==0 && ...
                           ~squareAttacked(board,8,5,'black') && ~squareAttacked(board,8,6,'black') && ~squareAttacked(board,8,7,'black')
                            addMove([8 5], [8 7], p, 0);
                        end
                        if st.wQc && board(8,4)==0 && board(8,3)==0 && board(8,2)==0 && ...
                           ~squareAttacked(board,8,5,'black') && ~squareAttacked(board,8,4,'black') && ~squareAttacked(board,8,3,'black')
                            addMove([8 5], [8 3], p, 0);
                        end
                    elseif friendSign == -1 && r == 1 && c == 5
                        % Black king on e8
                        if st.bKc && board(1,6)==0 && board(1,7)==0 && ...
                           ~squareAttacked(board,1,5,'white') && ~squareAttacked(board,1,6,'white') && ~squareAttacked(board,1,7,'white')
                            addMove([1 5], [1 7], p, 0);
                        end
                        if st.bQc && board(1,4)==0 && board(1,3)==0 && board(1,2)==0 && ...
                           ~squareAttacked(board,1,5,'white') && ~squareAttacked(board,1,4,'white') && ~squareAttacked(board,1,3,'white')
                            addMove([1 5], [1 3], p, 0);
                        end
                    end
            end
        end
        % Filter out any moves that would leave the moving side's king in check
        moves = filterSelfChecks(moves, st, side);
        
        function addMove(from, to, piece, capt)
            % Helper to add a move to the moves list
            mv.from = from;
            mv.to = to;
            mv.piece = piece;
            mv.captured = capt;
            moves(end+1) = mv;
        end
    end

    function validMoves = filterSelfChecks(candMoves, st, side)
        % Remove moves that result in own king in check
        validMoves = struct('from',{}, 'to',{}, 'piece',{}, 'captured',{});
        oppSide = opposite(side);
        for m = 1:numel(candMoves)
            mv = candMoves(m);
            newState = makeMove(st, mv);
            kingPos = findKing(newState.board, side);
            if isempty(kingPos), continue; end
            if ~squareAttacked(newState.board, kingPos(1), kingPos(2), oppSide)
                validMoves(end+1) = mv; %#ok<AGROW>
            end
        end
    end

    function stNew = makeMove(st, mv)
        % Execute move mv on state st, return the updated state
        stNew = st;
        from = mv.from;
        to = mv.to;
        piece = mv.piece;
        capt = mv.captured;
        % Handle en passant capture (pawn moving diagonally into empty square)
        if abs(piece) == 1 && st.board(to(1), to(2)) == 0 && from(2) ~= to(2)
            if piece > 0
                % White pawn captures black pawn en passant (remove pawn directly below target)
                stNew.board(to(1)+1, to(2)) = 0;
            else
                % Black pawn captures white pawn en passant (remove pawn directly above target)
                stNew.board(to(1)-1, to(2)) = 0;
            end
        end
        % Move the piece to the target square
        stNew.board(to(1), to(2)) = piece;
        stNew.board(from(1), from(2)) = 0;
        % Handle castling: move the rook as well
        if abs(piece) == 6 && abs(from(2) - to(2)) == 2
            if to(2) == 7  % king-side castle
                if piece > 0
                    % White: move rook from h1 (8,8) to f1 (8,6)
                    stNew.board(8,8) = 0;
                    stNew.board(8,6) = 4;
                else
                    % Black: move rook from h8 (1,8) to f8 (1,6)
                    stNew.board(1,8) = 0;
                    stNew.board(1,6) = -4;
                end
            elseif to(2) == 3  % queen-side castle
                if piece > 0
                    % White: move rook from a1 (8,1) to d1 (8,4)
                    stNew.board(8,1) = 0;
                    stNew.board(8,4) = 4;
                else
                    % Black: move rook from a8 (1,1) to d8 (1,4)
                    stNew.board(1,1) = 0;
                    stNew.board(1,4) = -4;
                end
            end
        end
        % Pawn promotion (automatically promote to Queen)
        if abs(piece) == 1
            if piece > 0 && to(1) == 1
                stNew.board(to(1), to(2)) = 5;   % white pawn -> queen
            elseif piece < 0 && to(1) == 8
                stNew.board(to(1), to(2)) = -5;  % black pawn -> queen
            end
        end
        % Update castling rights if king or rook moved or rook was captured
        if abs(piece) == 6
            % King moved: lose both castling rights for that side
            if piece > 0
                stNew.wKc = false; stNew.wQc = false;
            else
                stNew.bKc = false; stNew.bQc = false;
            end
        elseif abs(piece) == 4
            % Rook moved: lose specific castling right for that rook
            if piece > 0
                if from(1)==8 && from(2)==8, stNew.wKc = false; end
                if from(1)==8 && from(2)==1, stNew.wQc = false; end
            else
                if from(1)==1 && from(2)==8, stNew.bKc = false; end
                if from(1)==1 && from(2)==1, stNew.bQc = false; end
            end
        end
        if capt ~= 0 && abs(capt) == 4
            % A rook was captured: update that side's castling rights
            if capt > 0
                if to(1)==8 && to(2)==8, stNew.wKc = false; end
                if to(1)==8 && to(2)==1, stNew.wQc = false; end
            else
                if to(1)==1 && to(2)==8, stNew.bKc = false; end
                if to(1)==1 && to(2)==1, stNew.bQc = false; end
            end
        end
        % Update captured piece lists for display
        if capt ~= 0
            if capt > 0
                stNew.capturedWhite{end+1} = codeToChar(capt);
            else
                stNew.capturedBlack{end+1} = codeToChar(capt);
            end
        end
        % Store this move as the last move
        stNew.lastMove = mv;
    end

    function [over, mate] = isGameOver(st, side)
        % Determine if the game is over for the given side to move, and if it's checkmate or stalemate
        movesAvail = generateMoves(st, side);
        if isempty(movesAvail)
            over = true;
            oppSide = 'white'; 
            if strcmp(side,'white'), oppSide = 'black'; end
            kingPos = findKing(st.board, side);
            if ~isempty(kingPos) && squareAttacked(st.board, kingPos(1), kingPos(2), oppSide)
                mate = true;   % no moves and in check -> checkmate
            else
                mate = false;  % no moves and not in check -> stalemate
            end
        else
            over = false;
            mate = false;
        end
    end

    function [bestMove, bestScore] = search(st, key, depth, alpha, beta, side, ply)
        % Alpha-beta search with transposition table and quiescence search
        persistent TT TTinit;
        if isempty(TTinit)
            % Initialize transposition table (fixed size hash table)
            TT.size = 524287;
            TT.key   = zeros(1, TT.size, 'uint64');
            TT.score = zeros(1, TT.size, 'int32');
            TT.depth = zeros(1, TT.size, 'int8');
            TT.flag  = zeros(1, TT.size, 'int8');  % 0=exact, 1=lower bound, 2=upper bound
            TTinit = true;
        end
        bestMove = [];
        % Set maximizing/minimizing based on side (Black maximizes, White minimizes)
        localMaximize = strcmp(side, 'black');
        % Transposition table lookup
        idx = 1 + mod(double(key), TT.size);
        if TT.key(idx) == key && TT.depth(idx) >= depth
            % Use stored value if available at sufficient depth
            if TT.flag(idx) == 0
                bestScore = double(TT.score(idx));
                return;
            elseif TT.flag(idx) == 1 && TT.score(idx) >= beta
                bestScore = double(TT.score(idx));
                return;
            elseif TT.flag(idx) == 2 && TT.score(idx) <= alpha
                bestScore = double(TT.score(idx));
                return;
            end
        end
        if depth == 0
            % Reached leaf: perform quiescence search for stable evaluation
            bestScore = quiescence(st, key, alpha, beta, side);
            return;
        end
        % Generate all legal moves at this position
        localMoves = generateMoves(st, side);
        if isempty(localMoves)
            % No moves: checkmate or stalemate
            kingPos = findKing(st.board, side);
            if ~isempty(kingPos) && squareAttacked(st.board, kingPos(1), kingPos(2), opposite(side))
                % Checkmate for the side to move
                if strcmp(side,'white')
                    bestScore =  MATE_SCORE - ply;   % White to move and no moves in check -> Black wins
                else
                    bestScore = -MATE_SCORE + ply;   % Black to move and no moves in check -> White wins
                end
            else
                bestScore = 0;  % stalemate
            end
            return;
        end
        % Move ordering: consider captures first (MVV-LVA heuristic)
        localMoves = sortMovesMVVLV(localMoves);
        % Initialize bestScore for this node
        if localMaximize
            bestScore = -Inf;
        else
            bestScore = Inf;
        end
        % Save original alpha and beta for TT storage
        origAlpha = alpha;
        origBeta  = beta;
        for mv = localMoves
            % Recursively search child node
            childState = makeMove(st, mv);
            newKey = updateZobrist(st, key, mv);
            [~, score] = search(childState, newKey, depth-1, alpha, beta, opposite(side), ply+1);
            if localMaximize
                if score > bestScore
                    bestScore = score;
                    bestMove = mv;
                end
                alpha = max(alpha, bestScore);
            else
                if score < bestScore
                    bestScore = score;
                    bestMove = mv;
                end
                beta = min(beta, bestScore);
            end
            if alpha >= beta
                break;  % alpha-beta cutoff
            end
        end
        % Store the result in the transposition table
        TT.key(idx)   = key;
        TT.score(idx) = int32(bestScore);
        TT.depth(idx) = int8(depth);
        if bestScore <= origAlpha
            TT.flag(idx) = 2;   % upper bound
        elseif bestScore >= origBeta
            TT.flag(idx) = 1;   % lower bound
        else
            TT.flag(idx) = 0;   % exact score
        end
    end

    function score = quiescence(st, key, alpha, beta, side)
        % Quiescence search: evaluate only capture sequences to stabilize evaluation
        score = evaluatePosition(st);
        localMaximize = strcmp(side, 'black');
        if localMaximize
            if score >= beta, return; end
            alpha = max(alpha, score);
        else
            if score <= alpha, return; end
            beta = min(beta, score);
        end
        % Only consider capture moves in quiescence search
        moves = generateMoves(st, side);
        captures = moves([moves.captured] ~= 0);
        captures = sortMovesMVVLV(captures);
        for mv = captures
            childState = makeMove(st, mv);
            newKey = updateZobrist(st, key, mv);
            s = quiescence(childState, newKey, alpha, beta, opposite(side));
            if localMaximize
                if s > score, score = s; end
                alpha = max(alpha, score);
            else
                if s < score, score = s; end
                beta = min(beta, score);
            end
            if alpha >= beta
                break;
            end
        end
    end

    function val = evaluatePosition(st)
        % Static evaluation of the position, from Black's perspective (positive = advantage Black)
        % Material evaluation values
        valMap = [0, 100, 320, 330, 500, 900, 0];  % index: 0 unused, 1:P,2:N,3:B,4:R,5:Q,6:K
        board = st.board;
        % Material and positional evaluation
        materialScore = 0;
        positionalScoreWhite = 0;
        positionalScoreBlack = 0;
        [rows, cols, pieces] = find(board);
        for k = 1:numel(pieces)
            code = pieces(k);
            if code > 0
                materialScore = materialScore - valMap(code);
                positionalScoreWhite = positionalScoreWhite + pieceSquareValue(code, rows(k), cols(k));
            else
                materialScore = materialScore + valMap(abs(code));
                positionalScoreBlack = positionalScoreBlack + pieceSquareValue(code, rows(k), cols(k));
            end
        end
        val = materialScore + 0.1 * (positionalScoreBlack - positionalScoreWhite);
        % Additional positional heuristics
        % Bishop pair bonus
        if nnz(board == 3) >= 2, val = val - 50; end
        if nnz(board == -3) >= 2, val = val + 50; end
        % Rook placement (open/semi-open file bonuses)
        val = val + rookFileScore(board);
        % King safety and pawn structure
        val = val + kingSafety(board);
        val = val + pawnStructure(board);
        % Development: slight bonus for having more pieces (excluding pawns) in play
        val = val + 0.05 * (nnz(board < -1) - nnz(board > 1));
        % Neural network evaluation (if available)
        if useNN && ~isempty(net) && gammaNet ~= 0
            v = boardToVector(board);
            % If network was trained with normalization, apply it
            if ~isempty(st.Xnorm) && ~isempty(st.Ynorm)
                vNorm = mapminmax('apply', v, st.Xnorm);
                nnOut = net(vNorm);
                nnScore = mapminmax('reverse', nnOut, st.Ynorm);
            else
                nnScore = net(v);
            end
            % nnScore is from White's perspective; convert to Black perspective
            val = val - gammaNet * nnScore;
        end
    end

    %% Helper and Utility Functions %%
    function flag = squareAttacked(board, r, c, attackerSide)
        % Return true if square (r,c) is attacked by any piece of attackerSide
        flag = false;
        if strcmp(attackerSide, 'white')
            % White pawn attacks (one square diagonally up)
            if r < 8
                if c > 1 && board(r+1,c-1) == 1, flag = true; return; end
                if c < 8 && board(r+1,c+1) == 1, flag = true; return; end
            end
        else  % black attacker
            if r > 1
                if c > 1 && board(r-1,c-1) == -1, flag = true; return; end
                if c < 8 && board(r-1,c+1) == -1, flag = true; return; end
            end
        end
        % Knight attacks
        knightOffsets = [1 2; 1 -2; -1 2; -1 -2; 2 1; 2 -1; -2 1; -2 -1];
        for k = 1:8
            rr = r + knightOffsets(k,1);
            cc = c + knightOffsets(k,2);
            if rr < 1 || rr > 8 || cc < 1 || cc > 8, continue; end
            if strcmp(attackerSide,'white')
                if board(rr,cc) == 2, flag = true; return; end
            else
                if board(rr,cc) == -2, flag = true; return; end
            end
        end
        % Rook/Queen attacks (straight lines)
        directions = [1 0; -1 0; 0 1; 0 -1];
        for d = 1:4
            rr = r + directions(d,1);
            cc = c + directions(d,2);
            while rr >= 1 && rr <= 8 && cc >= 1 && cc <= 8
                if board(rr, cc) ~= 0
                    if strcmp(attackerSide,'white')
                        if board(rr,cc) == 4 || board(rr,cc) == 5, flag = true; return; end
                    else
                        if board(rr,cc) == -4 || board(rr,cc) == -5, flag = true; return; end
                    end
                    break;
                end
                rr = rr + directions(d,1);
                cc = cc + directions(d,2);
            end
        end
        % Bishop/Queen attacks (diagonals)
        directions = [1 1; 1 -1; -1 1; -1 -1];
        for d = 1:4
            rr = r + directions(d,1);
            cc = c + directions(d,2);
            while rr >= 1 && rr <= 8 && cc >= 1 && cc <= 8
                if board(rr, cc) ~= 0
                    if strcmp(attackerSide,'white')
                        if board(rr,cc) == 3 || board(rr,cc) == 5, flag = true; return; end
                    else
                        if board(rr,cc) == -3 || board(rr,cc) == -5, flag = true; return; end
                    end
                    break;
                end
                rr = rr + directions(d,1);
                cc = cc + directions(d,2);
            end
        end
        % King attacks (adjacent squares)
        for dr = -1:1
            for dc = -1:1
                if dr == 0 && dc == 0, continue; end
                rr = r + dr;
                cc = c + dc;
                if rr < 1 || rr > 8 || cc < 1 || cc > 8, continue; end
                if strcmp(attackerSide,'white')
                    if board(rr,cc) == 6, flag = true; return; end
                else
                    if board(rr,cc) == -6, flag = true; return; end
                end
            end
        end
    end

    function pos = findKing(board, side)
        % Find the coordinates [r,c] of the king of the given side
        if strcmp(side, 'white')
            [r, c] = find(board == 6);
        else
            [r, c] = find(board == -6);
        end
        if isempty(r)
            pos = [];
        else
            pos = [r(1), c(1)];
        end
    end

    function sortedMoves = sortMovesMVVLV(moves)
        % Sort moves by MVV-LVA heuristic (prioritize high-value captures)
        if isempty(moves)
            sortedMoves = moves;
            return;
        end
        % Assign simple values for piece types
        value = @(code) switch_abs_val(abs(code));
        function val = switch_abs_val(x)
            switch x
                case 1, val = 1;   % pawn
                case 2, val = 3;   % knight
                case 3, val = 3;   % bishop
                case 4, val = 5;   % rook
                case 5, val = 9;   % queen
                case 6, val = 10;  % king
                otherwise, val = 0;
            end
        end
        scores = zeros(1, numel(moves));
        for i = 1:numel(moves)
            if moves(i).captured ~= 0
                scores(i) = 10 * value(moves(i).captured) - value(moves(i).piece);
            end
        end
        [~, order] = sort(scores, 'descend');
        sortedMoves = moves(order);
    end

    function vec = boardToVector(board)
        % Convert the 8x8 board matrix into a 768x1 binary feature vector for NN input
        vec = zeros(768, 1);
        for sq = 1:64
            [rr, cc] = ind2sub([8 8], sq);
            piece = board(rr, cc);
            if piece ~= 0
                if piece > 0
                    idx = piece;              % White piece index 1-6
                else
                    idx = 6 + abs(piece);     % Black piece index 7-12
                end
                vec((idx-1)*64 + sq) = 1;
            end
        end
    end

    %-----------------------------------------------------------------------
% 1) LIGHT‑WEIGHT NETWORK FACTORY
%-----------------------------------------------------------------------
function net = createNetwork()
% Feed‑forward; fewer neurons; memory‑lean SCG trainer
%
%     768  →  [256 128 64 32]  →  1
%
% (You can of‑course change the layer widths.)

    net = feedforwardnet([256 128 64 32],'trainscg');

    % smaller initial weights → stabler incremental steps
    net.initFcn          = 'initlay';
    net.layers{end}.transferFcn = 'purelin';     % regression output
    for L = 1:numel(net.layers)-1
        net.layers{L}.transferFcn = 'tansig';
    end

    % no automatic data division – we feed mini‑batches manually
    net.divideFcn        = '';

    % modest defaults – we will override epochs per mini‑batch
    net.trainParam.epochs     = 3;
    net.trainParam.max_fail   = 6;
    net.trainParam.min_grad   = 0.0000001;
    net.trainParam.showWindow = false;
end


    function [X, Y] = parsePGNGame(gameText)
        % Parse a single PGN game text into training samples (input vectors X and target scores Y)
        X = []; Y = [];
        gameText = regexprep(gameText, '\[.*?\]', '');       % remove PGN tag pairs
        gameText = regexprep(gameText, '{[^}]*}', '');       % remove comments
        gameText = strrep(gameText, '...', ' ');
        gameText = regexprep(gameText, '\d+\.', '');         % remove move numbers
        gameText = strtrim(gameText);
        if isempty(gameText), return; end
        % Split moves by whitespace
        tokens = regexp(gameText, '\s+', 'split');
        simState = initState();
        sideToMove = 'white';
        for t = 1:length(tokens)
            moveStr = strtrim(tokens{t});
            if isempty(moveStr) || any(strcmp(moveStr, {'1-0','0-1','1/2-1/2','*'}))
                continue;  % skip result markers or empty tokens
            end
            mv = sanToMove(moveStr, simState, sideToMove);
            if isempty(mv)
                continue;  % skip unrecognized moves (if any)
            end
            simState = makeMove(simState, mv);
            % Randomly sample some positions to reduce sequential correlation
            if randi(4) == 1
                X = [X, boardToVector(simState.board)]; %#ok<AGROW>
                Y = [Y, simpleEval(simState.board)]; %#ok<AGROW>
            end
            % Switch side
            sideToMove = opposite(sideToMove);
        end
        % Ensure at least one sample per game (use final position if none collected)
        if isempty(X)
            X = boardToVector(simState.board);
            Y = simpleEval(simState.board);
        end
    end

    function games = splitPGN(pgnText)
        % Split a PGN file text into separate game strings
        pgnText = strrep(pgnText, '\r\n', '\n');
        pgnText = strrep(pgnText, '\r', '\n');
        games = regexp(pgnText, '\n\s*\n', 'split');
        games = games(~cellfun('isempty', games));
    end

    function mv = sanToMove(san, st, side)
        % Convert Standard Algebraic Notation (SAN) string to a move struct for the given state and side
        mv = struct('from',[],'to',[],'piece',[],'captured',[]);
        san = regexprep(strtrim(san), '[+#?!]*', '');  % remove annotations like +, #, etc.
        if isempty(san), return; end
        % Handle castling notation
        if strcmp(san, 'O-O') || strcmp(san, '0-0')
            if strcmp(side,'white')
                mv.from = [8 5]; mv.to = [8 7];
            else
                mv.from = [1 5]; mv.to = [1 7];
            end
            mv.piece = st.board(mv.from(1), mv.from(2));
            mv.captured = 0;
            return;
        elseif strcmp(san, 'O-O-O') || strcmp(san, '0-0-0')
            if strcmp(side,'white')
                mv.from = [8 5]; mv.to = [8 3];
            else
                mv.from = [1 5]; mv.to = [1 3];
            end
            mv.piece = st.board(mv.from(1), mv.from(2));
            mv.captured = 0;
            return;
        end
        % Handle promotion (e.g., e8=Q)
        promoPiece = '';
        if contains(san, '=')
            promoPiece = san(end);
            san = san(1:end-2);
        end
        % Determine moving piece type (default pawn if not specified)
        pieceChar = 'P';
        if any(san(1) == 'NBRQK')
            pieceChar = san(1);
            san(1) = [];
        end
        % Extract target square (file and rank)
        targetFile = san(end-1);
        targetRank = san(end);
        san(end-1:end) = [];
        if targetFile < 'a' || targetFile > 'h'
            return;  % invalid notation
        end
        destCol = double(targetFile) - double('a') + 1;
        destRow = 9 - str2double(targetRank);
        % Any remaining characters are disambiguation (file or rank of origin)
        disFile = 0; disRank = 0;
        for ch = san
            if isletter(ch)
                disFile = double(ch) - double('a') + 1;
            elseif ~isnan(str2double(ch))
                disRank = 9 - str2double(ch);
            end
        end
        % Find all candidate moves that match the SAN description
        candidates = generateMoves(st, side);
        for cand = candidates
            if upper(codeToChar(cand.piece)) == pieceChar && isequal(cand.to, [destRow, destCol])
                if (disFile == 0 || cand.from(2) == disFile) && (disRank == 0 || cand.from(1) == disRank)
                    mv = cand;
                    % Apply promotion piece if specified
                    if ~isempty(promoPiece) && abs(cand.piece) == 1
                        switch promoPiece
                            case 'Q', mv.piece = (cand.piece > 0) * 5 + (cand.piece < 0) * -5;
                            case 'R', mv.piece = (cand.piece > 0) * 4 + (cand.piece < 0) * -4;
                            case 'B', mv.piece = (cand.piece > 0) * 3 + (cand.piece < 0) * -3;
                            case 'N', mv.piece = (cand.piece > 0) * 2 + (cand.piece < 0) * -2;
                        end
                    end
                    return;
                end
            end
        end
        mv = [];  % if no candidate matched, return empty
    end

    function score = simpleEval(board)
        % Simple baseline evaluation: material balance from White's perspective
        pieceVals = [1, 3, 3, 5, 9, 0];  % P=1, N=3, B=3, R=5, Q=9, K=0
        score = 0;
        for v = board(:)'
            if v > 0
                score = score + pieceVals(v);
            elseif v < 0
                score = score - pieceVals(abs(v));
            end
        end
    end

    function score = rookFileScore(board)
        % Bonus for rooks on open or semi-open files
        score = 0;
        for file = 1:8
            hasWhitePawn = any(board(:,file) == 1);
            hasBlackPawn = any(board(:,file) == -1);
            if ~hasWhitePawn && ~hasBlackPawn
                % Open file
                if any(board(:,file) == 4),  score = score - 30; end
                if any(board(:,file) == -4), score = score + 30; end
            elseif ~hasWhitePawn && hasBlackPawn
                % Semi-open file for White (no white pawn)
                if any(board(:,file) == 4),  score = score - 15; end
            elseif hasWhitePawn && ~hasBlackPawn
                % Semi-open file for Black (no black pawn)
                if any(board(:,file) == -4), score = score + 15; end
            end
        end
    end

    function score = kingSafety(board)
        % Penalize having squares around the king under enemy attack
        score = 0;
        wKing = findKing(board, 'white');
        if ~isempty(wKing)
            attackers = 0;
            for dr = -1:1
                for dc = -1:1
                    if dr == 0 && dc == 0, continue; end
                    rr = wKing(1) + dr;
                    cc = wKing(2) + dc;
                    if rr >= 1 && rr <= 8 && cc >= 1 && cc <= 8
                        if squareAttacked(board, rr, cc, 'black')
                            attackers = attackers + 1;
                        end
                    end
                end
            end
            score = score + 10 * attackers;
        end
        bKing = findKing(board, 'black');
        if ~isempty(bKing)
            attackers = 0;
            for dr = -1:1
                for dc = -1:1
                    if dr == 0 && dc == 0, continue; end
                    rr = bKing(1) + dr;
                    cc = bKing(2) + dc;
                    if rr >= 1 && rr <= 8 && cc >= 1 && cc <= 8
                        if squareAttacked(board, rr, cc, 'white')
                            attackers = attackers + 1;
                        end
                    end
                end
            end
            score = score - 10 * attackers;
        end
    end

    function score = pawnStructure(board)
        % Evaluate pawn structure: doubled, isolated, and passed pawns
        score = 0;
        whitePawnCount = sum(board == 1);
        blackPawnCount = sum(board == -1);
        % Doubled pawns (each extra pawn on a file incurs a penalty)
        for file = 1:8
            if whitePawnCount(file) > 1
                score = score + 20 * (whitePawnCount(file) - 1);
            end
            if blackPawnCount(file) > 1
                score = score - 20 * (blackPawnCount(file) - 1);
            end
        end
        % Isolated pawns (no friendly pawn on adjacent files)
        for file = 1:8
            if whitePawnCount(file) > 0
                if (file == 1 || whitePawnCount(file-1) == 0) && (file == 8 || whitePawnCount(file+1) == 0)
                    score = score + 50 * whitePawnCount(file);
                end
            end
            if blackPawnCount(file) > 0
                if (file == 1 || blackPawnCount(file-1) == 0) && (file == 8 || blackPawnCount(file+1) == 0)
                    score = score - 50 * blackPawnCount(file);
                end
            end
        end
        % Passed pawns (no enemy pawn blocking them on the same file)
        [wr, wc] = find(board == 1);
        for i = 1:numel(wr)
            if ~any(board(1:wr(i)-1, wc(i)) == -1)
                score = score - 30;
            end
        end
        [br, bc] = find(board == -1);
        for i = 1:numel(br)
            if ~any(board(br(i)+1:8, bc(i)) == 1)
                score = score + 30;
            end
        end
    end

    function val = pieceSquareValue(piece, r, c)
        % Get piece-square table value for a given piece at position (r,c)
        persistent PST;
        if isempty(PST)
            % Piece-Square Tables (mid-game values)
            PST.pawn = [  0,  0,  0,  0,  0,  0,  0,  0;
                          50, 50, 50, 50, 50, 50, 50, 50;
                          10, 10, 20, 30, 30, 20, 10, 10;
                           5,  5, 10, 25, 25, 10,  5,  5;
                           0,  0,  0, 20, 20,  0,  0,  0;
                           5, -5,-10,  0,  0,-10, -5,  5;
                           5, 10, 10,-20,-20, 10, 10,  5;
                           0,  0,  0,  0,  0,  0,  0,  0 ];
            PST.knight = [ -50,-40,-30,-30,-30,-30,-40,-50;
                            -40,-20,  0,  5,  5,  0,-20,-40;
                            -30,  5, 10, 15, 15, 10,  5,-30;
                            -30,  0, 15, 20, 20, 15,  0,-30;
                            -30,  5, 15, 20, 20, 15,  5,-30;
                            -30,  0, 10, 15, 15, 10,  0,-30;
                            -40,-20,  0,  0,  0,  0,-20,-40;
                            -50,-40,-30,-30,-30,-30,-40,-50 ];
            PST.bishop = [ -20,-10,-10,-10,-10,-10,-10,-20;
                            -10,  5,  0,  0,  0,  0,  5,-10;
                            -10, 10, 10, 10, 10, 10, 10,-10;
                            -10,  0, 10, 10, 10, 10,  0,-10;
                            -10,  5,  5, 10, 10,  5,  5,-10;
                            -10,  0,  5, 10, 10,  5,  0,-10;
                            -10,  0,  0,  0,  0,  0,  0,-10;
                            -20,-10,-10,-10,-10,-10,-10,-20 ];
            PST.rook   = [   0,  0,  5, 10, 10,  5,  0,  0;
                             -5,  0,  0,  0,  0,  0,  0, -5;
                             -5,  0,  0,  0,  0,  0,  0, -5;
                             -5,  0,  0,  0,  0,  0,  0, -5;
                             -5,  0,  0,  0,  0,  0,  0, -5;
                             -5,  0,  0,  0,  0,  0,  0, -5;
                              5, 10, 10, 10, 10, 10, 10,  5;
                              0,  0,  0,  0,  0,  0,  0,  0 ];
            PST.queen  = [ -20,-10,-10, -5, -5,-10,-10,-20;
                            -10,  0,  0,  0,  0,  0,  0,-10;
                            -10,  0,  5,  5,  5,  5,  0,-10;
                             -5,  0,  5,  5,  5,  5,  0, -5;
                              0,  0,  5,  5,  5,  5,  0, -5;
                            -10,  5,  5,  5,  5,  5,  0,-10;
                            -10,  0,  5,  0,  0,  0,  0,-10;
                            -20,-10,-10, -5, -5,-10,-10,-20 ];
            PST.king   = [ -30,-40,-40,-50,-50,-40,-40,-30;
                            -30,-40,-40,-50,-50,-40,-40,-30;
                            -30,-40,-40,-50,-50,-40,-40,-30;
                            -30,-40,-40,-50,-50,-40,-40,-30;
                            -20,-30,-30,-40,-40,-30,-30,-20;
                            -10,-20,-20,-20,-20,-20,-20,-10;
                             20, 20,  0,  0,  0,  0, 20, 20;
                             20, 30, 10,  0,  0, 10, 30, 20 ];
        end
        switch abs(piece)
            case 1, table = PST.pawn;
            case 2, table = PST.knight;
            case 3, table = PST.bishop;
            case 4, table = PST.rook;
            case 5, table = PST.queen;
            case 6, table = PST.king;
        end
        if piece > 0
            % White piece: mirror table vertically (since White's perspective is from bottom)
            val = table(9-r, c);
        else
            % Black piece: use table as-is
            val = table(r, c);
        end
    end

    function key = zobristKey(board)
        % Compute a Zobrist hash key for the given board position
        persistent ZTable initialized;
        if isempty(initialized)
            rng(7);  % fixed seed for reproducibility
            ZTable = uint64(randi([0, 2^32-1], 12, 64));
            initialized = true;
        end
        key = uint64(0);
        for sq = 1:64
            [r, c] = ind2sub([8 8], sq);
            piece = board(r, c);
            if piece ~= 0
                if piece > 0
                    idx = piece;
                else
                    idx = 6 + abs(piece);
                end
                key = bitxor(key, ZTable(idx, sq));
            end
        end
    end

    function newKey = updateZobrist(st, oldKey, mv)
        % Update Zobrist hash key given a move mv from state st
        persistent ZTable initialized;
        if isempty(initialized)
            rng(7);
            ZTable = uint64(randi([0, 2^32-1], 12, 64));
            initialized = true;
        end
        newKey = oldKey;
        % Compute table indices for moving piece and possibly captured piece
        fromSq = sub2ind([8 8], mv.from(1), mv.from(2));
        toSq   = sub2ind([8 8], mv.to(1), mv.to(2));
        % Moving piece index
        if mv.piece > 0
            pIdx = mv.piece;
        else
            pIdx = 6 + abs(mv.piece);
        end
        % XOR out from square and XOR in to square
        newKey = bitxor(newKey, ZTable(pIdx, fromSq));
        newKey = bitxor(newKey, ZTable(pIdx, toSq));
        % If a piece was captured, XOR it out
        if mv.captured ~= 0
            if abs(mv.piece) == 1 && mv.from(2) ~= mv.to(2) && st.board(mv.to(1), mv.to(2)) == 0
                % En passant capture: captured pawn is not on 'to' square
                if mv.piece > 0
                    % White pawn captured a black pawn that was at (to(1)+1, to(2))
                    capSq = sub2ind([8 8], mv.to(1)+1, mv.to(2));
                    capIdx = 6 + 1;  % black pawn index
                else
                    % Black pawn captured a white pawn that was at (to(1)-1, to(2))
                    capSq = sub2ind([8 8], mv.to(1)-1, mv.to(2));
                    capIdx = 1;      % white pawn index
                end
            else
                capSq = toSq;
                if mv.captured > 0
                    capIdx = mv.captured;
                else
                    capIdx = 6 + abs(mv.captured);
                end
            end
            newKey = bitxor(newKey, ZTable(capIdx, capSq));
        end
    end

    function s = opposite(side)
        % Return the opposite side string ('white' <-> 'black')
        if strcmp(side, 'white')
            s = 'black';
        else
            s = 'white';
        end
    end

    function col = squareColor(isDarkSquare)
        % Return appropriate board square color
        if isDarkSquare
            col = colDark;
        else
            col = colLight;
        end
    end

    function glyph = pieceGlyph(code)
        % Map piece code to Unicode chess symbol for GUI display
        switch code
            case 1,  glyph = '♙';   % white pawn
            case 2,  glyph = '♘';   % white knight
            case 3,  glyph = '♗';   % white bishop
            case 4,  glyph = '♖';   % white rook
            case 5,  glyph = '♕';   % white queen
            case 6,  glyph = '♔';   % white king
            case -1, glyph = '♟';   % black pawn
            case -2, glyph = '♞';   % black knight
            case -3, glyph = '♝';   % black bishop
            case -4, glyph = '♜';   % black rook
            case -5, glyph = '♛';   % black queen
            case -6, glyph = '♚';   % black king
            otherwise, glyph = '';
        end
    end

    function char = codeToChar(code)
        % Map piece code to character (for notation and captured piece lists)
        switch code
            case 1,  char = 'P';
            case 2,  char = 'N';
            case 3,  char = 'B';
            case 4,  char = 'R';
            case 5,  char = 'Q';
            case 6,  char = 'K';
            case -1, char = 'p';
            case -2, char = 'n';
            case -3, char = 'b';
            case -4, char = 'r';
            case -5, char = 'q';
            case -6, char = 'k';
            otherwise, char = '';
        end
    end

    function cleanObj = removeHandles(obj)
        % Recursively remove graphics handles from an object (e.g., network) for safe saving
        if isstruct(obj)
            cleanObj = struct();
            fields = fieldnames(obj);
            for i = 1:numel(fields)
                cleanObj.(fields{i}) = removeHandles(obj.(fields{i}));
            end
        elseif iscell(obj)
            cleanObj = cell(size(obj));
            for i = 1:numel(obj)
                cleanObj{i} = removeHandles(obj{i});
            end
        else
            if ishghandle(obj)
                cleanObj = [];  % strip graphic handle
            else
                cleanObj = obj;
            end
        end
    end

end
    %--------------------------------------------------------------
    function code = charToCode(ch)
        % Map glyph-list character back to internal ±piece code
        switch ch
            case 'P', code =  1;
            case 'N', code =  2;
            case 'B', code =  3;
            case 'R', code =  4;
            case 'Q', code =  5;
            case 'K', code =  6;
            case 'p', code = -1;
            case 'n', code = -2;
            case 'b', code = -3;
            case 'r', code = -4;
            case 'q', code = -5;
            case 'k', code = -6;
            otherwise,  code = 0;
        end
    end
