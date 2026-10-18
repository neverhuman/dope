use super::*;
use std::net::TcpListener;
use std::thread;

fn parse(raw: &'static [u8]) -> Result<Request, &'static str> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let writer = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(raw).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
    });
    let (mut stream, _) = listener.accept().unwrap();
    let parsed = read_request(&mut stream);
    writer.join().unwrap();
    parsed
}

#[test]
fn rejects_chunked_and_non_json_mutations() {
    assert_eq!(
        Err("request_invalid"),
        parse(b"POST /v1/jobs HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n").map(|_| ())
    );
    assert_eq!(
        Err("request_invalid"),
        parse(b"POST /v1/jobs HTTP/1.1\r\nContent-Length: 0\r\nContent-Type: text/plain\r\n\r\n")
            .map(|_| ())
    );
}

#[test]
fn rejects_oversized_body_before_reading_it() {
    assert_eq!(
        Err("request_limit"),
        parse(b"POST /v1/jobs HTTP/1.1\r\nContent-Length: 65537\r\nContent-Type: application/json\r\n\r\n").map(|_| ())
    );
}
